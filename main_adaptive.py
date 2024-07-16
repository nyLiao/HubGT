import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
import os
import time
import argparse
from tqdm import tqdm
from torch.nn.functional import normalize
import scipy.sparse as sp
from numpy.linalg import inv

from model import GT
from preprocess_data import process_data
from utils.collator import collator
from utils.lr import PolynomialDecayLR


def get_time(device=None):
    torch.cuda.synchronize(device=device)
    return time.time()


def train(args, model, device, loader, optimizer, lr_scheduler):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y_true = batch.y.view(-1)
        loss = F.nll_loss(pred, y_true, ignore_index=-1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_train(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss_list.append(F.nll_loss(pred, batch.y.view(-1), ignore_index=-1).item())
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))

    y_pred = torch.cat(y_pred).reshape(-1)
    y_true = torch.cat(y_true).reshape(-1)
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return acc, np.mean(loss_list)


def eval(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss_list.append(F.nll_loss(pred, batch.y.view(-1), ignore_index=-1).item())
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    pred_list = []
    for i in torch.split(y_pred, args.num_data_augment, dim=0):
        pred_list.append(i.bincount().argmax().unsqueeze(0))
    y_pred = torch.cat(pred_list)
    y_true = y_true.view(-1, args.num_data_augment)[:, 0]
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return acc, np.mean(loss_list)


def random_split(data_list, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(len(data_list))
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * len(data_list))]
    val_idx = all_idx[int(frac_train * len(data_list)):int((frac_train+frac_valid) * len(data_list))]
    test_idx = all_idx[int((frac_train+frac_valid) * len(data_list)):]
    train_list = []
    test_list = []
    val_list = []
    for i in train_idx:
        train_list.append(data_list[i])
    for i in val_idx:
        val_list.append(data_list[i])
    for i in test_idx:
        test_list.append(data_list[i])
    return train_list, val_list, test_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph transformer')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('-d', '--data', type=str, default='cora', help='Dataset name')
    parser.add_argument('--data_split', type=str, default='60/20/20', help='Index or percentage of dataset split')
    parser.add_argument('--normg', type=float, default=0.5, help='Generalized graph norm')
    parser.add_argument('--normf', type=int, nargs='?', default=0, const=None, help='Embedding norm dimension. 0: feat-wise, 1: node-wise, None: disable')
    parser.add_argument('--multi', action='store_true', help='True for multi-label classification')
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_data_augment', type=int, default=8)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('-v', '--device', type=int, default=3, help='which gpu to use if any (default: 0)')
    parser.add_argument('--perturb_feature', type=bool, default=False)
    parser.add_argument('--weight_update_period', type=int, default=10000, help='epochs to update the sampling weight')
    parser.add_argument('--K', type=int, default=16, help='use top-K shortest path distance as feature')
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    data_list, feature, y = process_data(args)
    # if not os.path.exists('./dataset/' + args.data):
    #     data_list, feature, y = process_data(args)
    # else:
    #     data_list = torch.load('./dataset/'+args.data+'/data.pt')
    #     feature = torch.load('./dataset/'+args.data+'/feature.pt')
    #     y = torch.load('./dataset/'+args.data+'/y.pt')

    if args.data in ['arxiv-year', 'snap-patents']:
        frac_train, frac_valid, frac_test = 0.5, 0.25, 0.25
    elif args.data in ['squirrel', 'chameleon', 'wisconsin']:
        frac_train, frac_valid, frac_test = 0.48, 0.32, 0.20
    else:
        frac_train, frac_valid, frac_test = 0.6, 0.2, 0.2
    train_dataset, test_dataset, valid_dataset = random_split(data_list, frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test, seed=args.seed)
    print('dataset load successfully')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, shuffle=True, perturb=args.perturb_feature))
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, shuffle=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, shuffle=False))
    print(args)

    model = GT(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=feature.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=y.max().item()+1,
        attn_bias_dim=args.attn_bias_dim,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        ffn_dim=args.ffn_dim,
        num_global_node=args.num_global_node
    )
    if not args.test and not args.validate:
        print(model)
    print('Total params:', sum(p.numel() for p in model.parameters()))
    model.to(device)
    print('device:', device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_epochs,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)

    val_acc_list = []

    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, lr_scheduler)
        lr_scheduler.step()

        start = get_time(device)
        train_acc, train_loss = eval_train(args, model, device, train_loader)
        epoch_time = get_time(device) - start

        val_acc, val_loss = eval(args, model, device, val_loader)

        print_str = f'Epoch: {epoch:02d}, ' + \
                    f'train_loss: {train_loss:.4f}, ' + \
                    f'train_acc: {train_acc * 100:.2f}%, ' + \
                    f'val_loss: {val_loss:.2f}, ' + \
                    f'val_acc: {val_acc * 100:.2f}%, ' + \
                    f'Time: {epoch_time:.2f}s'
        print(print_str)
        val_acc_list.append(val_acc)

    test_acc, test_loss = eval(args, model, device, test_loader)
    print('best validation acc: ', max(val_acc_list))
    print('best test acc: ', test_acc, test_loss)

if __name__ == "__main__":
    main()
