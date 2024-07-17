import os
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
from utils.config import setup_argparse, setup_args
from utils.collator import collate
from Precompute import PyPLL


def choice_cap(a: list, size: int, p: np.ndarray=None):
    if len(a) <= size:
        return a
    p = p / np.sum(p)
    return np.random.choice(a, size, replace=False, p=p).tolist()


def process_data(args):
    # TODO: address LargestConnectedComponents by using s_total global landmarks
    data_loader = SingleGraphLoader(args)
    data, metric = data_loader(args)
    num_nodes = data.num_nodes
    x, y = data.x, data.y
    undirected = pyg_utils.is_undirected(data.edge_index)
    deg = pyg_utils.degree(data.edge_index[0], num_nodes, dtype=int)
    id_map = torch.argsort(deg, descending=False)
    id_map_inv = torch.empty_like(id_map)
    id_map_inv[id_map] = torch.arange(num_nodes)

    py_pll = PyPLL()
    edge_index = data.edge_index.numpy().astype(np.uint32)
    time_pre = py_pll.construct_index(edge_index, args.kindex, not undirected)
    del edge_index, data.edge_index
    data.edge_index = None

    s_total = args.ss
    sw0, sw1, sw_spd = utils.Stopwatch(), utils.Stopwatch(), utils.Stopwatch()
    pw0, pw1, pw2 = utils.Stopwatch(), utils.Stopwatch(), utils.Stopwatch()
    indices = torch.empty((2, 0), dtype=int)
    ids = torch.zeros((num_nodes, args.ns, s_total), dtype=int)
    for iego, ego in enumerate(tqdm(id_map)):
        pw0.start()
        with sw0:
            nodes0, val0, s0_actual = py_pll.label(ego)
            val0 = np.array(val0) ** args.r0
        s0 = min(args.s0, s0_actual)
        s1 = s_total - s0 - 1
        with sw1:
            nodes1, val1, s1_actual = py_pll.s_neighbor(ego, s1)
            val1 = np.array(val1) ** args.r1
            # nodes1, val1, s1_actual = py_pll.s_push(ego, s1, args.r1)
            # TODO: reduce dulp for s_push
        pw0.pause()
        pw1.start()

        indices_i = torch.empty((2, 0), dtype=int)
        for s in range(args.ns):
            subgraph = torch.empty((s_total,), dtype=int)
            subgraph[0] = ego

            # Add label nodes
            g0 = choice_cap(nodes0, args.s0, val0)
            subgraph[1:len(g0)+1] = torch.tensor(g0, dtype=int)

            # Add neighbor nodes
            g1 = choice_cap(nodes1, s1, val1)
            subgraph[len(g0)+1:] = torch.tensor(g1, dtype=int)

            ids[ego, s] = subgraph
            subgraph = torch.unique(subgraph)
            ui, vi = torch.triu_indices(subgraph.size(0), subgraph.size(0), offset=1)
            u, v = subgraph[ui], subgraph[vi]
            # TODO: edge dropout
            ii = torch.stack([u, v], dim=0)
            indices_i = torch.cat([indices_i, ii], dim=1)
        mask = (indices_i[0] < indices_i[1])
        indices_i = indices_i[:, mask]
        indices_i = pyg_utils.coalesce(indices_i)
        indices = torch.cat([indices, indices_i], dim=1)
        pw1.pause()

    pw2.start()
    mask = (indices[0] < indices[1])
    indices = indices[:, mask]
    indices = pyg_utils.coalesce(indices, num_nodes=num_nodes)
    values  = torch.empty_like(indices[0], dtype=torch.int16)
    for i, (u, v) in enumerate(tqdm(indices.t())):
        with sw_spd:
            values[i] = py_pll.k_distance_query(u, v, 1)[0]
    pw2.pause()

    spd = sp.coo_matrix(
        (values.numpy(), indices.numpy()),
        shape=(num_nodes, num_nodes),
        dtype=np.int16
    ).tocsr()
    graph = Data(x=x, y=y, num_nodes=num_nodes, spd=spd)
    graph.contiguous('x', 'y')
    # os.makedirs('./dataset/' + args.data, exist_ok=True)
    # torch.save(graph, './dataset/'+args.data+'/spd.pt')
    # torch.save(ids, './dataset/'+args.data+'/ids.pt')
    print(pw0, pw1, pw2)
    print(f'Labeling time: {sw0}, Neighbor time: {sw1}, SPD time: {sw_spd}, Size: {graph.num_edges}')

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
    print(s)

    return loader


if __name__ == '__main__':
    args = setup_args(setup_argparse())
    process_data(args)
