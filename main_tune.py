import logging
import optuna
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F

from model import GT
from preprocess import process_data
import utils
from utils import (
    Accumulator, Stopwatch,
    ResLogger, CkptLogger,
    PolynomialDecayLR
)
from main import learn, eval


def objective(trial, args, logger, res_logger):
    args = deepcopy(args)
    res_logger = deepcopy(res_logger)
    args.perturb_std = trial.suggest_float('perturb_std', 0.0, 0.05, step=0.001)
    args.kfeat = trial.suggest_categorical('kfeat', [0, 4, 8])
    args.ns = trial.suggest_int('ns', 2, 8, step=2)
    args.num_global_node = trial.suggest_int('num_global_node', 0, 1)
    args.s0 = trial.suggest_int('s0', 0, 22, step=2)
    args.s1 = trial.suggest_int('s1', 0, 8, step=2)
    args.r0 = trial.suggest_float('r0', -2.0, 2.0, step=0.1)
    args.r1 = trial.suggest_float('r1', -2.0, 2.0, step=0.1)
    for k in trial.params:
        res_logger.concat([(k, trial.params[k])])
    logger.log(logging.LTRN, trial.params)

    # ========== Load data
    args.ss = args.ss - args.num_global_node
    args.quiet = True
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
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        ffn_dim=args.ffn_dim,
        num_global_node=args.num_global_node
    )
    logger.log(logging.LTRN, str(model))
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(),
            lr=args.peak_lr,
            weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.epoch // 10,
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
    logger.log(logging.LTRN, f'Total params: {utils.ParamNumel(model)(unit='K')} K')

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

        trial.report(metric_val, epoch-1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    res_logger.concat(ckpt_logger.get_at_best())
    res_logger.concat([
        ('epoch', ckpt_logger.epoch_current),
        ('time', time_learn.data),
    ], suffix='learn')

    # ========== Run testing
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
    return res_logger.data.loc[0, args.metric+'_val']


def main(args):
    # ========== Run configuration
    logger = utils.setup_logger(args.logpath, level_console=args.loglevel, level_file=30, quiet=args.quiet)
    res_logger = ResLogger(args.logpath, prefix='param', quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    study_path = '-'.join(filter(None, ['optuna', args.suffix])) + '.db'
    study_path, _ = utils.setup_logpath(folder_args=(study_path,))
    study = optuna.create_study(
        study_name=args.logid,
        storage=f'sqlite:///{str(study_path)}',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=2,
            max_resource=args.epoch,
            reduction_factor=3),
        load_if_exists=True)
    optuna.logging.set_verbosity(args.loglevel)

    study.optimize(
        partial(objective, args=args, logger=logger, res_logger=res_logger),
        n_trials=args.n_trials,
        # gc_after_trial=True,
        show_progress_bar=True,)
    best_params = {k: v for k, v in study.best_params.items()}
    utils.save_args(args.logpath, best_params)
    utils.clear_logger(logger)


if __name__ == "__main__":
    parser = utils.setup_argparse()
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    args = utils.setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = utils.setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}-param'
        args.logpath, args.logid = utils.setup_logpath(
            folder_args=(args.data, args.flag),
            quiet=args.quiet)

        main(args)
