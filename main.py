import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import GT
from preprocess import process_data
import utils
from utils.lr import PolynomialDecayLR


def learn(model, device, loader, optimizer):
    model.train()
    loss_epoch = utils.Accumulator()
    stopwatch = utils.Stopwatch()

    for batch in loader:
        batch = batch.to(device)
        with stopwatch:
            optimizer.zero_grad()
            output = model(batch)
            label  = batch.y.view(-1)
            loss = F.nll_loss(output, label, ignore_index=-1)
            loss.backward()
            optimizer.step()

        loss_epoch.update(loss.item(), count=label.size(0))

    return utils.ResLogger()(
        [('time_learn', stopwatch.data),
         (f'loss_{loader.split}', loss_epoch.mean * loader.batch * loader.ns)])


# @torch.no_grad()
# def eval(model, device, loader, evaluator):
#     model.eval()
#     stopwatch = utils.Stopwatch()

#     for batch in loader:
#         batch = batch.to(device)
#         with stopwatch:
#             output = model(batch)
#             label  = batch.y.view(-1)
#         evaluator(output.argmax(1), label)

#     res = utils.ResLogger()
#     res.concat(evaluator.compute())
#     res.concat([('time', stopwatch.data)], suffix=loader.split)
#     evaluator.reset()
#     stopwatch.reset()
#     return res


def eval(model, device, loader, evaluator=None):
    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    pred_list = []
    for i in torch.split(y_pred, loader.ns, dim=0):
        pred_list.append(i.bincount().argmax().unsqueeze(0))
    y_pred = torch.cat(pred_list)
    y_true = y_true.view(-1, loader.ns)[:, 0]
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return utils.ResLogger().concat([('f1_micro', acc)])


def main(args):
    # ========== Run configuration
    logger = utils.setup_logger(args.logpath, level_console=args.loglevel, quiet=args.quiet)
    res_logger = utils.ResLogger(quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    # ========== Load data
    loader = process_data(args)

    # ========== Load model
    model = GT(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=args.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        attn_bias_dim=1,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        ffn_dim=args.ffn_dim,
        num_global_node=args.num_global_node
    )
    print(model)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(),
            lr=args.peak_lr,
            weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_epochs,
            tot=args.epoch,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    evaluator = utils.F1Calculator(args.num_classes)
    ckpt_logger = utils.CkptLogger(
            args.logpath,
            patience=args.patience,
            period=1,
            prefix=('-'.join(filter(None, ['model', args.suffix]))),)
    print(args)

    # ========== Run training
    logger.debug('-'*20 + f" Start training: {args.epoch} " + '-'*20)
    time_learn = utils.Accumulator()
    res_learn = utils.ResLogger()
    for epoch in range(1, args.epoch+1):
        res_learn.concat([('epoch', epoch, lambda x: format(x, '03d'))], row=epoch)

        res = learn(model, args.device, loader['train'], optimizer)
        res_learn.merge(res, rows=[epoch])
        time_learn.update(res_learn[epoch, 'time_learn'])

        res = eval(model, args.device, loader['val'], evaluator)
        res_learn.merge(res, rows=[epoch])
        metric_val = res_learn[epoch, 'f1_micro']
        scheduler.step()

        logger.log(logging.LTRN, res_learn.get_str(row=epoch))

        ckpt_logger.step(metric_val, model)
        ckpt_logger.set_at_best(epoch_best=epoch)
        if ckpt_logger.is_early_stop:
            break

    res_train = utils.ResLogger()
    res_train.concat(ckpt_logger.get_at_best())
    res_train.concat(
        [('epoch', ckpt_logger.epoch_current),
            ('time', time_learn.data),],
        suffix='learn')
    print(res_train)

    # ========== Run testing
    model = ckpt_logger.load('best', model=model)
    res_test = eval(model, args.device, loader['test'], evaluator)
    print(res_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=utils.force_list_int, default=[42], help='random seed')
    parser.add_argument('-v', '--device', type=int, default=3, help='which gpu to use if any (default: 0)')
    parser.add_argument('-z', '--suffix', type=str, default=None, help='Save name suffix.')
    parser.add_argument('--loglevel', type=int, default=10, help='10:progress, 15:train, 20:info, 25:result')
    parser.add_argument('-quiet', action='store_true', help='Dry run without saving logs.')
    # Model configuration
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_global_node', type=int, default=1)
    # Optim configuration
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('-e', '--epoch', type=int, default=1000)
    parser.add_argument('-p', '--patience', type=int, default=50, help='Patience epoch for early stopping')
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    # Data configuration
    parser.add_argument('-d', '--data', type=str, default='citeseer', help='Dataset name')
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('--data_split', type=str, default='60/20/20', help='Index or percentage of dataset split')
    parser.add_argument('--multi', action='store_true', help='True for multi-label classification')
    parser.add_argument('--kindex', type=int, default=8, help='top-K PLL')
    parser.add_argument('-ns', type=int, default=8, help='num of subgraphs')
    parser.add_argument('-ss', type=int, default=31, help='total num of nodes in each subgraph')
    parser.add_argument('-s0', type=int, default=15, help='max num of label nodes in each subgraph')
    parser.add_argument('-r0', type=float, default=-1.0, help='norm for label distance')
    parser.add_argument('-r1', type=float, default=-1.0, help='norm for neighbor distance')
    parser = utils.setup_argparse(parser)
    args = utils.setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = utils.setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'
        args.logpath, args.logid = utils.setup_logpath(
            folder_args=(args.data, 'GT', args.flag),
            quiet=args.quiet)

        main(args)
