# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
"""
import os
import json
import uuid
import random
import argparse
from pathlib import Path

import numpy as np
import torch


# noinspection PyUnresolvedReferences
def setup_seed(seed: int = None, cuda: bool = True) -> int:
    if seed is None:
        seed = int(uuid.uuid4().hex, 16) % 1000000
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    return seed


def setup_cuda(args: argparse.Namespace) -> argparse.Namespace:
    args.cuda = args.device >= 0 and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args


def setup_argparse():
    np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                        formatter=dict(float=lambda x: f"{x: 9.3e}"))
    torch.set_printoptions(linewidth=160, edgeitems=5)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=force_list_int, default=[42], help='random seed')
    parser.add_argument('-v', '--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('-z', '--suffix', type=str, default=None, help='Save name suffix.')
    parser.add_argument('--loglevel', type=int, default=10, help='10:progress, 15:train, 20:info, 25:result')
    parser.add_argument('-quiet', action='store_true', help='Dry run without saving logs.')
    parser.add_argument('-index', action='store_true', help='Force saving index.')
    # Model configuration
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--dp_input', type=float, default=0.1)
    parser.add_argument('--dp_bias', type=float, default=0.1)
    parser.add_argument('--dp_ffn', type=float, default=0.5)
    parser.add_argument('--dp_attn', type=float, default=0.5)
    parser.add_argument('--aggr_output', type=int, default=0)       # bool
    parser.add_argument('--var_vfeat', type=int, default=1)         # bool
    # Optim configuration
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('-e', '--epoch', type=int, default=200)
    parser.add_argument('-p', '--patience', type=int, default=20, help='Patience epoch for early stopping')
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    # Data configuration
    parser.add_argument('-d', '--data', type=str, default='citeseer', help='Dataset name')
    parser.add_argument('-b', '--batch', type=int, default=1024)
    parser.add_argument('--data_split', type=str, default='60/20/20', help='Index or percentage of dataset split')
    parser.add_argument('--multi', action='store_true', help='True for multi-label classification')
    parser.add_argument('--num_workers', type=int, default=8, help='number of loader workers')
    parser.add_argument('--perturb_std', type=float, default=0.0, help='perturb for training data')
    parser.add_argument('--pre_collate', type=int, default=1)       # bool
    # Precompute configuration
    parser.add_argument('--kindex', type=int, default=8, help='top-K PLL indexing')
    parser.add_argument('--kbias', type=int, default=1, help='top-K SPD for bias')
    parser.add_argument('--kfeat', type=int, default=0, help='top-K SPD for feature')
    parser.add_argument('-ns', type=int, default=1, help='num of subgraphs')
    parser.add_argument('-ss', type=int, default=48, help='total num of nodes in each subgraph')
    parser.add_argument('-s0', type=int, default=24, help='max num of label nodes in each subgraph')
    parser.add_argument('-s0g', type=int, default=8, help='max num of global nodes in each subgraph')
    parser.add_argument('-s1', type=int, default=12, help='max num of rev label nodes in each subgraph')
    parser.add_argument('-r0', type=float, default=-1.0, help='norm for label distance')
    parser.add_argument('-r0g', type=float, default=-2.0, help='norm for global distance')
    parser.add_argument('-r1', type=float, default=-1.0, help='norm for neighbor distance')
    return parser


def setup_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Check args
    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()
    args = setup_cuda(args)
    return args


def save_args(logpath: Path, args: dict):
    if 'quiet' in args and args['quiet']:
        return
    with open(logpath.joinpath('config.json'), 'w') as f:
        f.write(json.dumps(dict_to_json(args), indent=4))


def dict_to_json(dictionary) -> dict:
    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except:
            return False

    filtered_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            filtered_value = dict_to_json(value)
        elif isinstance(value, list):
            filtered_value = [v for v in value if is_serializable(v)]
        elif is_serializable(value):
            filtered_value = value
        else:
            try:
                filtered_value = str(value)
            except:
                continue
        filtered_dict[key] = filtered_value
    return filtered_dict


force_list_str = lambda x: [str(v) for v in x.split(',')]
force_list_int = lambda x: [int(v) for v in x.split(',')]
list_str = lambda x: [str(v) for v in x.split(',')] if isinstance(x, str) and ',' in x else str(x)
list_int = lambda x: [int(v) for v in x.split(',')] if isinstance(x, str) and ',' in x else int(x)
list_float = lambda x: [float(v) for v in x.split(',')] if isinstance(x, str) and ',' in x else float(x)
