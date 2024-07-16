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


def setup_argparse(parser):
    np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                        formatter=dict(float=lambda x: f"{x: 9.3e}"))
    torch.set_printoptions(linewidth=160, edgeitems=5)
    return parser


def setup_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Check args
    args = parser.parse_args()
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
