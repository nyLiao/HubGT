import argparse
import numpy as np

import torch_geometric.utils as pyg_utils

from load_data import SingleGraphLoader
from Precompute import PyPLL


def process_data(args, use_coarsen_feature=True):
    data_loader = SingleGraphLoader(args)
    data, metric = data_loader(args)
    undirected = pyg_utils.is_undirected(data.edge_index)
    edge_index = data.edge_index.numpy().astype(np.uint32)

    py_pll = PyPLL()
    t_pre = py_pll.construct_index(edge_index, args.K, not undirected)
    del edge_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('-v', '--dev', type=int, default=0, help='GPU id')
    # Data configuration
    parser.add_argument('-d', '--data', type=str, default='cora', help='Dataset name')
    parser.add_argument('--data_split', type=str, default='60/20/20', help='Index or percentage of dataset split')
    parser.add_argument('--normg', type=float, default=0.5, help='Generalized graph norm')
    parser.add_argument('--normf', type=int, nargs='?', default=0, const=None, help='Embedding norm dimension. 0: feat-wise, 1: node-wise, None: disable')
    parser.add_argument('--multi', action='store_true', help='True for multi-label classification')

    parser.add_argument('-K', type=int, default=16, help='use top-K shortest path distance as feature')
    args = parser.parse_args()
    process_data(args)
