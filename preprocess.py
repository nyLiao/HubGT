import os
import gc
import logging
from tqdm import tqdm
from functools import partial
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

from load_data import SingleGraphLoader
import utils
from utils.collator import collate, INF8
from Precompute2 import PyPLL

N_BPROOT = 128


def choice_cap(a: list, size: int, nsample: int, p: np.ndarray=None):
    if size == 0:
        return
    if len(a) <= size:
        return np.tile(a, (nsample, 1))
    p = p / np.sum(p)
    ret = [np.random.choice(a, size, replace=False, p=p) for _ in range(nsample)]
    return np.vstack(ret)


def callback_sample(ids, ret):
    ego, ids_i = ret
    ids[ego] = ids_i


def aggr_sample(ego, ziplst, **args):
    # Sample neighbors
    ids_i = np.full((args['ns'], args['ss']), ego, dtype=np.int32)
    s_top = 1
    for nodes, val, r, s in ziplst:
        if s == 0:
            continue
        val = np.asarray(val) ** r
        val = np.nan_to_num(val, posinf=0)
        ids_i[:, s_top:s_top+s] = choice_cap(nodes, s, args['ns'], val)
        s_top += s
    return ego, ids_i


def aggr_csr(ids_chunk, num_nodes):
    # Dense connection in each subgraph
    spd = sp.csr_matrix((num_nodes, num_nodes), dtype=int,)
    for ids_i in ids_chunk:
        indices_i = []
        for ids_s in ids_i:
            subset = np.unique(ids_s)
            s_row, s_col = np.triu_indices(len(subset), 1)
            s_row, s_col = subset[s_row], subset[s_col]
            indices_i.append(np.vstack([s_row, s_col]))
        indices_i = np.hstack(indices_i)
        mask = (indices_i[0] < indices_i[1])
        indices_i = indices_i[:, mask]
        values_i = np.ones_like(indices_i[0], dtype=int)
        spd += sp.coo_matrix(
            (values_i, indices_i),
            shape=(num_nodes, num_nodes),
        ).tocsr()
    return spd


def process_data(args, res_logger=utils.ResLogger()):
    logger = logging.getLogger('log')
    data_loader = SingleGraphLoader(args)
    data, metric = data_loader(args)
    res_logger.concat([
        ('data', args.data),
        ('metric', metric),
    ])
    num_nodes = data.num_nodes
    x, y = data.x, data.y
    undirected = pyg_utils.is_undirected(data.edge_index)
    edge_index = data.edge_index.numpy().astype(np.uint32)
    if not undirected:
        logger.warning(f'Warning: Directed graph.')

    deg = pyg_utils.degree(data.edge_index[0], num_nodes, dtype=int)
    id_map = torch.argsort(deg, descending=False).numpy().astype(np.uint32)
    deg_max = deg[id_map[-1]]

    # ===== Build label
    py_pll = PyPLL()
    # time_index = py_pll.construct_index(edge_index, args.kindex, not undirected, args.quiet)
    time_index = py_pll.get_index(
        edge_index, np.flip(id_map), str(args.logpath.parent), args.quiet, args.index)
    del edge_index, data.edge_index, deg
    data.edge_index = None

    stopwatch_sample, stopwatch_spd = utils.Stopwatch(), utils.Stopwatch()
    ids = np.zeros((num_nodes, args.ns, args.ss), dtype=int)
    fn_callback = partial(callback_sample, ids)
    n1_lst = {e: [] for e in range(num_nodes)}
    pool = Pool(args.num_workers)
    stopwatch_sample.start()
    for ego in id_map:
        # ===== Generate SPD neighborhood
        # TODO: fix directed API
        nodes0, val0, s0_actual = py_pll.label(ego)
        if args.s1 > 0:
            for node, val in zip(nodes0, val0):
                n1_lst[node].append((ego, val))
        s0 = min(args.s0, s0_actual)
        ziplst = [(nodes0, val0, args.r0, s0)] if s0 > 0 else []

        nodes0g, val0g, s0g_actual = py_pll.glabel(ego)
        s0g = min(args.s0g, s0g_actual)
        if s0g > 0:
            ziplst.append((nodes0g, val0g, args.r0, s0g))

        s1_actual = len(n1_lst[ego])
        if s1_actual > 0:
            nodes1, val1 = zip(*n1_lst[ego])
            nodes1 = list(nodes1)
            s1_actual = min(args.s1, len(nodes1))
            ziplst.append((nodes1, val1, args.r0, s1_actual))
        s1 = s1_actual
        del n1_lst[ego]

        s2 = args.ss - s0 - s0g - s1 - 1
        s2_actual = 0
        if s2 > 0:
            nodes2, val2, s2_actual = py_pll.s_neighbor(ego, s2)
            s2_actual = min(s2_actual, s2)
            ziplst.append((nodes2, val2, args.r1, s2_actual))
        s2 = s2_actual

        kwargs = {'ns': args.ns, 'ss': args.ss}
        pool.apply_async(aggr_sample, (ego, ziplst), kwargs, callback=fn_callback)
    pool.close()
    pool.join()
    stopwatch_sample.pause()

    # ===== Aggregate SPD graph
    id_map = np.array_split(np.random.permutation(num_nodes), args.num_workers)
    with Pool(args.num_workers) as pool:
        spd = pool.starmap(aggr_csr, ((ids[id_i], num_nodes+N_BPROOT) for id_i in id_map))

    spd = sum(spd)
    spd.sum_duplicates()
    rows, cols, _ = sp.find(spd)
    rows, cols = rows.astype(np.uint32), cols.astype(np.uint32)
    spd.data = np.arange(len(rows), dtype=int)
    with stopwatch_spd:
        spd_bias = py_pll.k_distance_parallel(rows, cols)
        # spd_bias = py_pll.k_distance_parallel(rows, cols, args.kfeat)
    spd_bias = torch.tensor(spd_bias, dtype=int).view(-1, args.kbias)
    logger.log(logging.LTRN, f'SPD size: {spd.nnz}, feat size: {x.size(1)}, max deg: {deg_max}')
    gc.collect()

    # ===== Extend features
    y = torch.cat([y, -torch.ones(N_BPROOT, dtype=y.dtype)])
    x = torch.cat([x, torch.zeros(N_BPROOT, x.size(1), dtype=x.dtype)], dim=0)
    if args.kfeat > 0:
        rows = cols = np.arange(x.size(0), dtype=np.uint32)
        with stopwatch_spd:
            x_extend = py_pll.k_distance_parallel(rows, cols, args.kfeat)
        x_extend = torch.tensor(x_extend, dtype=torch.float32).view(-1, args.kfeat)
        x_extend = (INF8 - x_extend) / INF8
        x = torch.cat([x, x_extend], dim=1)
        args.num_features = x.size(1)
    logger.log(logging.LTRN, f'Indexing time: {time_index:.2f}, Neighbor time: {stopwatch_sample}, SPD time: {stopwatch_spd}')

    # ===== Data loader
    graph = Data(x=x, y=y, num_nodes=num_nodes, spd=spd.tocsr(), spd_bias=spd_bias)
    graph.contiguous('x', 'y', 'spd_bias')

    s = ''
    loader = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        shuffle = {'train': True, 'val': False, 'test': False}[split]
        std = {'train': args.perturb_std, 'val': 0.0, 'test': 0.0}[split]
        loader[split] = DataLoader(
            pyg_utils.mask_to_index(mask),
            batch_size=args.batch,
            num_workers=args.num_workers,
            shuffle=shuffle,
            collate_fn=partial(collate,
                ids=torch.from_numpy(ids),
                graph=graph,
                std=std,
        ))
        s += f'{split}: {mask.sum().item()}, '
    logger.log(logging.LTRN, s)

    res_logger.concat([
        ('time_index', time_index),
        ('time_pre', time_index + stopwatch_sample.data + stopwatch_spd.data),
        ('mem_ram_pre', utils.MemoryRAM()(unit='G')),
        ('mem_cuda_pre', utils.MemoryCUDA()(unit='G')),
    ])
    return loader


if __name__ == '__main__':
    logging.LTRN = 15
    args = utils.setup_args(utils.setup_argparse())
    args.logpath, args.logid = utils.setup_logpath(
        folder_args=(args.data, str(args.seed[0])),
        quiet=args.quiet)
    np.random.seed(args.seed[0])
    logger = utils.setup_logger(level_file=30, quiet=args.quiet)
    process_data(args)
