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
from utils.collator import INF8, collate_fetch
from Precompute2 import PyPLL

N_BPROOT = 128
np.set_printoptions(linewidth=160, edgeitems=50, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=50)


def choice_cap(a: list, size: int, nsample: int, p: np.ndarray=None):
    if size == 0:
        return
    if len(a) <= size:
        return np.tile(a, (nsample, 1))
    p = p / np.sum(p)
    ret = [np.random.choice(a, size, replace=False, p=p) for _ in range(nsample)]
    return np.vstack(ret)


def callback_sample(ids, pbar, ret):
    pbar.update(1)
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
    spd.sum_duplicates()
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
    args.num_nodes = num_nodes
    x, y = data.x, data.y
    undirected = pyg_utils.is_undirected(data.edge_index)
    edge_index = data.edge_index.numpy().astype(np.uint32)
    if not undirected:
        logger.warning(f'Warning: Directed graph.')

    deg = np.bincount(edge_index[0], minlength=num_nodes)
    id_map = np.argsort(deg, kind='stable').astype(np.uint32)

    # ===== Build label
    py_pll = PyPLL()
    # time_index = py_pll.construct_index(edge_index, args.kindex, not undirected, args.quiet)
    time_index = py_pll.get_index(
        edge_index, np.flip(id_map), str(args.logpath.parent), args.index, args.quiet)
    py_pll.set_args(args.quiet, args.seed[0], args.ss, args.s0g, args.s0, args.s1)
    logger.log(logging.LTRN, f'Index time: {time_index:.2f}')
    del edge_index, data.edge_index, deg
    data.edge_index = None

    # ===== Extend features
    y = torch.cat([y, -torch.ones(N_BPROOT, dtype=y.dtype)])
    x = torch.cat([x, torch.zeros(N_BPROOT, x.size(1), dtype=x.dtype)], dim=0)

    # ===== Data loader
    s = ''
    loader = {}
    graph = Data(x=x, y=y, num_nodes=num_nodes)
    graph.contiguous('x', 'y')

    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        shuffle = {'train': True, 'val': False, 'test': False}[split]
        std = {'train': args.perturb_std, 'val': 0.0, 'test': 0.0}[split]
        loader[split] = DataLoader(
            pyg_utils.mask_to_index(mask),
            batch_size=args.batch,
            num_workers=(0 if num_nodes < 5e4 else 4),
            shuffle=shuffle,
            collate_fn=partial(collate_fetch,
                c_handler=py_pll, graph=graph, s_total=args.ss, std=std,)
        )
        s += f'{split}: {mask.sum().item()}, '
    logger.log(logging.LTRN, s)
    gc.collect()

    res_logger.concat([
        ('time_index', time_index),
        ('time_pre', time_index),
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

    loader = process_data(args)
    with utils.Stopwatch() as timer:
        next(iter(loader['val']))
    logger.log(logging.LTRN, f'Batch time: {timer}')
    with utils.Stopwatch() as timer:
        for batch in loader['val']:
            pass
    logger.log(logging.LTRN, f'Full time: {timer}')
