import argparse

from load_data import SingleGraphLoader


def process_data(args, use_coarsen_feature=True):
    data_loader = SingleGraphLoader(args)
    data, metric = data_loader(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data configuration
    parser.add_argument('-d', '--data', type=str, default='cora', help='Dataset name')
    parser.add_argument('--data_split', type=str, default='60/20/20', help='Index or percentage of dataset split')
    parser.add_argument('--normg', type=float, default=0.5, help='Generalized graph norm')
    parser.add_argument('--normf', type=int, nargs='?', default=0, const=None, help='Embedding norm dimension. 0: feat-wise, 1: node-wise, None: disable')
    parser.add_argument('--multi', action='store_true', help='True for multi-label classification')

    parser.add_argument('--K', type=int, default=16, help='use top-K shortest path distance as feature')
    args = parser.parse_args()
    process_data(args)
