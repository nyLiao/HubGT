import os
import uuid
import logging
import optuna
from copy import deepcopy
from functools import partial

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
from main import learn, eval


def objective(trial, args, logger, res_logger):
    res_logger = deepcopy(res_logger)
    args = deepcopy(args)
    # args.perturb_std = trial.suggest_float('perturb_std', 0.0, 0.05, step=0.001)
    args.aggr_output = trial.suggest_int('aggr_output', 0, 1)
    args.var_vfeat = trial.suggest_int('var_vfeat', 0, 1)
    # args.kfeat = trial.suggest_int('kfeat', 0, 8, step=4)
    # args.ns = trial.suggest_int('ns', 2, 8, step=2)
    args.s0 = trial.suggest_int('s0', 0, 24, step=2)
    args.s0g = trial.suggest_int('s0g', 0, 10, step=2)
    args.s1 = trial.suggest_int('s1', 0, 12, step=2)
    # args.r0 = trial.suggest_float('r0', -4.0, 2.0, step=0.2)
    # args.r0g = trial.suggest_float('r0g', -4.0, 2.0, step=0.2)
    # args.r1 = trial.suggest_float('r1', -4.0, 2.0, step=0.2)
    for k in trial.params:
        res_logger.concat([(k, trial.params[k])])
    logger.log(logging.LTRN, trial.params)

    # ========== Load data
    if os.path.exists('./cache/'+args.data+'/index.bin'):
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
    trial.set_user_attr("s_test", res_logger._get(col=args.metric+'_test', row=0))
    return res_logger.data.loc[0, args.metric+'_val']


def main(args):
    # ========== Run configuration
    logger = utils.setup_logger(args.logpath, level_console=args.loglevel, level_file=30, quiet=args.quiet)
    res_logger = ResLogger(args.logpath, prefix='param', quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    study_path = '-'.join(filter(None, ['optuna', args.suffix])) + '.db'
    study_path, _ = utils.setup_logpath(folder_args=(study_path,))
    study_name = '/'.join(str(args.logpath).split('/')[-2:-1])
    study_id = '/'.join((study_name, str(args.seed)))
    study_id = uuid.uuid5(uuid.NAMESPACE_DNS, study_id).int % 2**32
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna.storages.RDBStorage(
            url=f'sqlite:///{str(study_path)}',
            heartbeat_interval=3600),
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=min(6, args.n_trials // 5),
            n_ei_candidates=24,
            seed=study_id,
            multivariate=True,
            group=True,
            warn_independent_sampling=False),
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
    args.n_trials = args.n_trials // len(seed_lst)
    for seed in seed_lst:
        args.seed = utils.setup_seed(seed, args.cuda)
        args.flag = f'param'
        args.logpath, args.logid = utils.setup_logpath(
            folder_args=(args.data, args.flag),
            quiet=args.quiet)

        main(args)
