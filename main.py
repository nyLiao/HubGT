import os
import gc
import uuid
import json
import logging
import optuna

import torch
import torch.nn.functional as F

from model import GT
from preprocess import process_data, N_BPROOT
import utils
from utils import (
    Accumulator, Stopwatch,
    ResLogger, CkptLogger,
    PolynomialDecayLR,
)


def learn(args, model, device, loader, optimizer):
    model.train()
    loss_epoch = Accumulator()
    stopwatch = Stopwatch()
    criterion = F.binary_cross_entropy_with_logits if (args.num_classes == 1 or args.multi) else F.cross_entropy

    for batch in loader:
        batch = batch.to(device)
        with stopwatch:
            optimizer.zero_grad()
            output = model(batch)
            label  = batch.y.view(-1)
            if args.num_classes == 1:
                label = label.unsqueeze(1).float()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        loss_epoch.update(loss.item(), count=label.size(0)/args.ns)

    return ResLogger()(
        [('time_learn', stopwatch.data),
         ('loss_train', loss_epoch.mean)])


@torch.no_grad()
def eval(args, model, device, loader, evaluator):
    model.eval()
    stopwatch = Stopwatch()

    for batch in loader:
        batch = batch.to(device)
        with stopwatch:
            output = model(batch)
            label  = batch.y.view(-1, args.ns)[:, 0]

        if args.num_classes == 1:
            output = F.sigmoid(output.squeeze(-1).view(-1, args.ns))
            pred = (output > 0.5).long()
            mask = torch.mode(pred, dim=1).values
            mask = torch.where(pred == mask[:, None], 1, 0)
            output = torch.sum(output * mask, dim=1) / mask.sum(dim=1)
        else:
            # Average output logits on subgraph dim based on majority predicted class
            output = F.log_softmax(output, dim=1).view(-1, args.ns, args.num_classes)
            pred = output.argmax(dim=2)  # [batch, ns]
            mask = torch.mode(pred, dim=1).values
            mask = torch.where(pred == mask[:, None], 1, 0)
            output = torch.sum(output * mask[:, :, None], dim=1) / mask.sum(dim=1)[:, None]
        evaluator(output, label)

    res = ResLogger()
    res.concat(evaluator.compute())
    res.concat([('time_eval', stopwatch.data)])
    evaluator.reset()
    stopwatch.reset()
    return res


def main(args):
    # ========== Run configuration
    logger = utils.setup_logger(args.logpath, level_console=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(quiet=args.quiet, suffix=args.suffix)
    res_logger.concat([('seed', args.seed),])

    # ========== Load data
    loader = process_data(args, res_logger)
    with args.device:
        torch.cuda.empty_cache()

    # ========== Load model
    logger.debug('-'*20 + f" Loading model " + '-'*20)
    model = GT(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=args.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        attn_bias_dim=args.kbias,
        dp_attn=args.dp_attn,
        dp_ffn=args.dp_ffn,
        dp_input=args.dp_input,
        dp_bias=args.dp_bias,
        ffn_dim=args.ffn_dim,
        num_nodes=args.num_nodes,
        num_global_node=N_BPROOT,
        var_vfeat=bool(args.var_vfeat),
        aggr_output=bool(args.aggr_output),
    )
    logger.log(logging.LTRN, str(model))
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(),
            lr=args.peak_lr,
            weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(
            optimizer,
            warmup=max(args.epoch // 10, 50),
            tot=args.epoch,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    evaluator = utils.get_evaluator(args).to(args.device)
    evaluator = {split: evaluator.clone(postfix='_'+split) for split in ['val', 'test']}
    ckpt_logger = CkptLogger(
            args.logpath,
            patience=args.patience,
            period=1,
            prefix=('-'.join(filter(None, ['model', args.suffix]))),)
    logger.log(logging.LTRN, f'Total params: {utils.ParamNumel(model)(unit="K")} K')

    # ========== Run training
    logger.debug('-'*20 + f" Start training: {args.epoch} " + '-'*20)
    time_learn = Accumulator()
    res_learn = ResLogger()
    for epoch in range(1, args.epoch+1):
        res_learn.concat([('epoch', epoch, lambda x: format(x, '03d'))], row=epoch)

        res = learn(args, model, args.device, loader['train'], optimizer)
        res_learn.merge(res, rows=[epoch])
        time_learn.update(res_learn[epoch, 'time_learn'])

        res = eval(args, model, args.device, loader['val'], evaluator['val'])
        res_learn.merge(res, rows=[epoch])
        metric_val = res_learn[epoch, args.metric+'_val']
        scheduler.step()

        logger.log(logging.LTRN, res_learn.get_str(row=epoch))

        ckpt_logger.step(metric_val, model)
        ckpt_logger.set_at_best(**{'epoch_best': epoch, args.metric+'_val': metric_val})
        if ckpt_logger.is_early_stop:
            break

    res_logger.concat(ckpt_logger.get_at_best())
    res_logger.concat([
        ('epoch', ckpt_logger.epoch_current),
        ('time', time_learn.data),
    ], suffix='learn')

    # ========== Run testing
    del loader['train'], loader['val']
    gc.collect()
    model = ckpt_logger.load('best', model=model)
    res_test = eval(args, model, args.device, loader['test'], evaluator['test'])
    res_logger.merge(res_test)
    res_logger.concat([
        ('mem_ram_learn', utils.MemoryRAM()(unit='G')),
        ('mem_cuda_learn', utils.MemoryCUDA()(unit='G')),
    ])

    logger.info(f"[args]: {args}")
    logger.log(logging.LRES, f"[res]: {res_logger}")
    res_logger.save()
    utils.save_args(args.logpath, vars(args))
    utils.clear_logger(logger)
    return res_logger.data.loc[0, args.metric+'_val']


if __name__ == "__main__":
    parser = utils.setup_argparse()
    parser.add_argument('--seed_tune', type=int, default=None, help='Load seed')
    args = utils.setup_args(parser)

    if args.seed_tune is not None:
        logpath, logid = utils.setup_logpath(
            folder_args=(args.data, 'param'),
            quiet=True)
        if os.path.exists(logpath.joinpath('config.json')):
            with open(logpath.joinpath('config.json'), 'r') as config_file:
                best_params = json.load(config_file)
        else:
            study_path, _ = utils.setup_logpath(folder_args=('optuna.db',))
            study_name = '/'.join(str(args.logpath).split('/')[-2:-1])
            study_id = '/'.join((study_name, str(args.seed)))
            study_id = uuid.uuid5(uuid.NAMESPACE_DNS, study_id).int % 2**32
            print(f"Saving to {logpath}.")
            study = optuna.create_study(
                study_name=study_id,
                storage=f'sqlite:///{str(study_path)}',
                direction='maximize',
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=2,
                    max_resource=args.epoch,
                    reduction_factor=3),
                load_if_exists=True)
            best_params = {k: v for k, v in study.best_params.items()}
            utils.save_args(logpath, best_params)

        for k, v in best_params.items():
            setattr(args, k, v)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = utils.setup_seed(seed, args.cuda)
        args.flag = '-'.join(filter(None, [str(args.seed), args.suffix]))
        args.logpath, args.logid = utils.setup_logpath(
            folder_args=(args.data, args.flag),
            quiet=args.quiet)

        main(args)
