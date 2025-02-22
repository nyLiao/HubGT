import os
import gc
import logging
from tqdm import tqdm
from functools import partial
import numpy as np

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
    seed = args.seed if isinstance(args.seed, int) else args.seed[0]
    quiet = (args.loglevel > 10)
    py_pll.set_args(quiet, seed, args.ss, args.s0g, args.s0, args.s1)
    # time_index = py_pll.construct_index(edge_index, args.kindex, not undirected, args.quiet)
    time_index = py_pll.get_index(
        edge_index, np.flip(id_map), str(args.logpath.parent), args.index, args.quiet)
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
            num_workers=(1 if num_nodes < 5e4 else args.num_workers),
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
        batch = next(iter(loader['val']))
    # print(batch.ids.numpy()[0, :])
    # print(batch.attn_bias.numpy()[0, :, :, 0])
    logger.log(logging.LTRN, f'Batch time: {timer}')
    with utils.Stopwatch() as timer:
        for batch in loader['val']:
            pass
    logger.log(logging.LTRN, f'Full time: {timer}')

# Avg Label size: 2.28088 + 0.599573 + 0.783166
# Avg 2-hop size: 3.30724 (0, 0.322819, 0.320093, 0.161215, 0.118769, 0.0771028, 0)
# Index time: 0.03
# [   3 1409 3172 1409 3172   64 1220 1526    3 1409]
# [[0 1 1 1 1 2 2 3 0 1]
#  [1 0 2 0 2 1 1 2 1 0]
#  [1 2 0 2 0 3 1 2 1 2]
#  [1 0 2 0 2 1 1 2 1 0]
#  [1 2 0 2 0 3 1 2 1 2]
#  [2 1 3 1 3 0 2 3 2 1]
#  [2 1 1 1 1 2 0 1 2 1]
#  [3 2 2 2 2 3 1 0 3 2]
#  [0 1 1 1 1 2 2 3 0 1]
#  [1 0 2 0 2 1 1 2 1 0]]
