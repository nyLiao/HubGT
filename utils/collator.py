import random
import torch
from torch.utils.data import TensorDataset
import torch_geometric.utils as pyg_utils

from .efficiency import Stopwatch

INF8 = 127


def pad_1d_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch(object):
    def __init__(self, attn_bias, x, y, ids=None):
        super(Batch, self).__init__()
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        # self.ids = ids

    def to(self, device):
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias = self.attn_bias.to(device)
        # self.ids = self.ids.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def collator(items, feature, shuffle=False, perturb=False):
    batch_list = []
    for item in items:  # batch nodes
        for x in item:  # sample subgraph
            batch_list.append((x[0], x[1], x[2][0]))
    if shuffle:
        random.shuffle(batch_list)
    # attn_biases: (B*Sample, n_subgraph, n_subgraph, hop)
    # xs(ids): (B*Sample, n_subgraph), ys: (B*Sample)
    attn_biases, xs, ys = zip(*batch_list)
    max_node_num = max(i.size(0) for i in xs)
    y = torch.cat([i.unsqueeze(0) for i in ys])
    # x: (B*Sample, n_subgraph, feature_dim)
    x = torch.cat([pad_2d_unsqueeze(feature[i], max_node_num) for i in xs])
    ids = torch.cat([i.unsqueeze(0) for i in xs])
    if perturb:
        x += torch.FloatTensor(x.shape).uniform_(-0.1, 0.1)
    attn_bias = torch.cat([i.unsqueeze(0) for i in attn_biases])

    # top-k shortest path feature
    # dis = []
    # for subgraph in xs:
    #     u = subgraph[0].item()
    #     for v in subgraph:
    #         v = v.item()
    #         if (u, v) in kspd or (v, u) in kspd:
    #             dis.append(kspd[(u, v)] if (u, v) in kspd else kspd[(v, u)])
    #         else:
    #             dis.append(torch.zeros((K, )))
    # dis = torch.cat(dis).reshape(x.shape[0], x.shape[1], -1)
    # x = torch.cat([x, dis], dim=-1)

    return Batch(
        attn_bias=attn_bias,
        x=x,
        y=y,
        ids=ids,
    )


class NodeDataLoader(object):
    def __init__(self, ids, spd, split, args) -> None:
        self.ids = ids
        self.spd = spd

        self.split = split
        self.shuffle = {'train': True, 'val': False, 'test': False}[split]
        self.perturb = {'train': False, 'val': False, 'test': False}[split]
        self.batch = args.batch
        self.ns = args.ns
        self.device = args.device

    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(len(self.ids))
        else:
            idxs = torch.arange(len(self.ids))
        self.idxs = idxs.split(self.batch)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.idxs):
            raise StopIteration
        idx = self.idxs[self.current]
        n_seq = idx.size(0) * self.ns
        self.current += 1

        ids = self.ids[idx].view(n_seq, -1)     # [batch_size * ns, s_total]
        s_total = ids.size(1)
        y = self.spd.y[ids[:, 0].view(-1)].view(-1)

        attn_bias = torch.empty((n_seq, s_total, s_total),
            dtype=torch.int,
            device=self.device
        )
        for g, subnodes in enumerate(ids):
            subset, inv_id = torch.unique(subnodes, return_inverse=True)
            # index_new is the occurrence of index_old in subset
            edge_index, edge_attr = pyg_utils.subgraph(
                subset,
                self.spd.edge_index,
                self.spd.edge_attr,
                relabel_nodes=True,
                num_nodes=self.spd.num_nodes,
            )
            edge_index, edge_attr = pyg_utils.to_undirected(
                edge_index,
                edge_attr,
                reduce="min"
            )

            spd_g = pyg_utils.to_dense_adj(
                edge_index,
                edge_attr=edge_attr.type(torch.int),
            )[0]
            mask = ~torch.eye(spd_g.size(0), dtype=torch.bool) & (spd_g <= 0)
            spd_g[mask] = INF8
            attn_bias[g] = spd_g[inv_id][:, inv_id]

        x = self.spd.x[ids.view(-1)].view(n_seq, s_total, -1)
        if self.perturb:
            x += torch.normal(0, 0.05, x.shape, device=x.device, dtype=x.dtype)
        return Batch(attn_bias, x, y)

    def __len__(self):
        return len(self.idxs)
