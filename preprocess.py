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
from utils.collator import collate, INF8
from Precompute import PyPLL


def choice_cap(a: list, size: int, nsample: int, p: np.ndarray=None):
    if len(a) <= size:
        return np.tile(a, (nsample, 1))
    p = p / np.sum(p)
    ret = [np.random.choice(a, size, replace=False, p=p) for _ in range(nsample)]
    return np.vstack(ret)


def process_data(args, res_logger):
    # TODO: address LargestConnectedComponents by using s_total global landmarks
    data_loader = SingleGraphLoader(args)
    data, metric = data_loader(args)
    res_logger.concat([
        ('data', args.data),
        ('metric', metric),
    ])
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
    spd = sp.coo_matrix((num_nodes, num_nodes), dtype=int,)
    ids = torch.zeros((num_nodes, args.ns, s_total), dtype=int)
    for iego, ego in enumerate(tqdm(id_map)):
        # Generate SPD neighborhood
        pw0.start()
        with sw0:
            nodes0, val0, s0_actual = py_pll.label(ego)
            val0 = np.array(val0) ** args.r0
            val0[val0 == np.inf] = INF8 - 1
        s0 = min(args.s0, s0_actual)
        s1 = s_total - s0 - 1
        with sw1:
            nodes1, val1, s1_actual = py_pll.s_neighbor(ego, s1)
            val1 = np.array(val1) ** args.r1
            # nodes1, val1, s1_actual = py_pll.s_push(ego, s1, args.r1)
            # TODO: reduce dulp for s_push
        pw0.pause()
        pw1.start()

        # Sample neighbors
        indices_i = np.zeros((2, args.ns * s_total), dtype=np.int16)
        ids_i = [
            np.repeat(ego, args.ns).reshape(-1, 1),
            choice_cap(nodes0, args.s0, args.ns, val0),
            choice_cap(nodes1, s1, args.ns, val1)
        ]
        ids_i = np.hstack(ids_i)      # [ns, s_total]
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
        spd_i = sp.coo_matrix(
            (values_i, indices_i),
            shape=(num_nodes, num_nodes),
        )
        spd += spd_i
        pw1.pause()

    pw2.start()
    spd.sum_duplicates()
    rows, cols, _ = sp.find(spd)
    for i, (u, v) in enumerate(tqdm(zip(rows, cols), total=len(rows))):
        with sw_spd:
            spd.data[i] = py_pll.k_distance_query(u, v, 1)[0]
    # spd.data = spd.data.clip(0, INF8)
    # spd.data = (spd.data * (float(INF8) / np.max(spd.data))).astype(int)
    pw2.pause()

    graph = Data(x=x, y=y, num_nodes=num_nodes, spd=spd.tocsr())
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

    res_logger.concat([
        ('mem_ram_pre', utils.MemoryRAM()(unit='G')),
        ('mem_cuda_pre', utils.MemoryCUDA()(unit='G')),
    ])
    return loader


if __name__ == '__main__':
    args = utils.setup_args(utils.setup_argparse())
    process_data(args)
