import os
import logging
from tqdm import tqdm
from functools import partial
import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

from load_data import SingleGraphLoader
import utils
from utils.collator import collate, INF8
from Precompute import PyPLL

logging.LTRN = 15


def choice_cap(a: list, size: int, nsample: int, p: np.ndarray=None):
    if len(a) <= size:
        return np.tile(a, (nsample, 1))
    p = p / np.sum(p)
    ret = [np.random.choice(a, size, replace=False, p=p) for _ in range(nsample)]
    return np.vstack(ret)


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
    if not undirected:
        logger.warning(f'Warning: Directed graph.')

    deg = pyg_utils.degree(data.edge_index[0], num_nodes, dtype=int)
    id_map = torch.argsort(deg, descending=False)
    id_map_inv = torch.empty_like(id_map)
    id_map_inv[id_map] = torch.arange(num_nodes)

    py_pll = PyPLL()
    edge_index = data.edge_index.numpy().astype(np.uint32)
    time_pre = py_pll.construct_index(edge_index, args.kindex, not undirected)
    del edge_index, data.edge_index
    data.edge_index = None

    stopwatch_sample, stopwatch_spd = utils.Stopwatch(), utils.Stopwatch()
    s_total = args.ss
    spd = sp.coo_matrix((num_nodes, num_nodes), dtype=int,)
    ids = torch.zeros((num_nodes, args.ns, s_total), dtype=int)
    n1_lst = {e: [] for e in range(num_nodes)}
    # TODO: parallelize
    for iego, ego in enumerate(tqdm(id_map, disable=args.quiet)):
        stopwatch_sample.start()
        # Generate SPD neighborhood
        # TODO: fix directed API
        nodes0, val0, s0_actual = py_pll.label(ego)
        val0 = np.array(val0) ** args.r0
        val0[val0 == np.inf] = 0
        if args.s1 > 0:
            for node, val in zip(nodes0, val0):
                n1_lst[node].append((int(ego), val))
        s0 = min(args.s0, s0_actual)

        s1_actual = len(n1_lst[int(ego)])
        if s1_actual > 0:
            nodes1, val1 = zip(*n1_lst[int(ego)])
            nodes1 = list(nodes1)
            val1 = np.array(val1) ** args.r0
        s1 = min(args.s1, s1_actual)

        s2 = s_total - s0 - s1 - 1
        s2_actual = 0
        if s2 > 0:
            nodes2, val2, s2_actual = py_pll.s_neighbor(ego, s2)
            val2 = np.array(val2) ** args.r1
        s2 = min(s2, s2_actual)

        # Sample neighbors
        ids_i = np.full((args.ns, s_total), ego, dtype=np.int16)
        ids_i[:, 1:s0+1] = choice_cap(nodes0, s0, args.ns, val0)
        if s1 > 0:
            ids_i[:, s0+1:s0+s1+1] = choice_cap(nodes1, s1, args.ns, val1)
        if s2 > 0:
            ids_i[:, s0+s1+1:s0+s1+s2+1] = choice_cap(nodes2, s2, args.ns, val2)
        ids[ego] = torch.tensor(ids_i, dtype=int)

        # Append SPD indices
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
        )
        stopwatch_sample.pause()

    # SPD graph value
    spd.sum_duplicates()
    rows, cols, _ = sp.find(spd)
    spd_bias = torch.zeros((len(rows), args.kbias), dtype=torch.int16)
    # for i, (u, v) in enumerate(tqdm(zip(rows, cols), total=len(rows))):
    for i, (u, v) in enumerate(zip(rows, cols)):
        stopwatch_spd.start()
        spd.data[i] = i
        kspd = py_pll.k_distance_query(u, v, args.kbias)
        if len(kspd) > 0:
            spd_bias[i] = torch.tensor(kspd, dtype=int)
        stopwatch_spd.pause()

    if args.kfeat > 0:
        x_extend = torch.empty((x.size(0), args.kfeat), dtype=x.dtype).fill_(INF8)
        for i in range(x.size(0)):
            stopwatch_spd.start()
            kspd = py_pll.k_distance_query(i, i, args.kfeat)
            if len(kspd) > 0:
                x_extend[i, :len(kspd)] = torch.tensor(kspd, dtype=x.dtype)
            stopwatch_spd.pause()
        x_extend = (INF8 - x_extend) / INF8
        x = torch.cat([x, x_extend], dim=1)
        args.num_features = x.size(1)

    graph = Data(x=x, y=y, num_nodes=num_nodes, spd=spd.tocsr(), spd_bias=spd_bias)
    graph.contiguous('x', 'y', 'spd_bias')
    # os.makedirs('./dataset/' + args.data, exist_ok=True)
    # torch.save(graph, './dataset/'+args.data+'/graph.pt')
    # torch.save(ids, './dataset/'+args.data+'/ids.pt')

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
                ids=ids,
                graph=graph,
                std=std,
        ))
        s += f'{split}: {mask.sum().item()}, '
    logger.log(logging.LTRN, s)
    logger.log(logging.LTRN, f'SPD size: {spd.nnz}, feat size: {x.size(1)}')
    logger.log(logging.LTRN, f'Indexing time: {time_pre:.2f}, Neighbor time: {stopwatch_sample}, SPD time: {stopwatch_spd}')

    res_logger.concat([
        ('time_pre', time_pre + stopwatch_sample.data + stopwatch_spd.data),
        ('mem_ram_pre', utils.MemoryRAM()(unit='G')),
        ('mem_cuda_pre', utils.MemoryCUDA()(unit='G')),
    ])
    return loader


if __name__ == '__main__':
    args = utils.setup_args(utils.setup_argparse())
    process_data(args)
