import torch

INF8 = 127


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


def gen_attn_bias(graph, subnodes, device='cpu') -> torch.Tensor:
    def slice_subgraph(edge_index, edge_attr, subset, num_nodes):
        node_mask = subset.new_zeros(num_nodes, dtype=torch.bool)
        node_mask[subset] = True
        edge_mask = node_mask[edge_index[0]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask]
        edge_mask = node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask]
        return edge_index, edge_attr

    # Construct sparse subgraph
    subset, inv_id = torch.unique(subnodes, return_inverse=True)
    edge_index, edge_attr = slice_subgraph(
        graph.edge_index, graph.edge_attr, subset, graph.num_nodes)
    edge_index, edge_attr, subset = edge_index.to(device), edge_attr.to(device), subset.to(device)

    # Relabel nodes
    relabel = lambda x: (x.view(-1, 1) == subset).int().argmax(dim=1)
    edge_index = relabel(edge_index).view(2, -1)

    # Convert to dense matrix
    spd_g = torch.zeros(subset.size(0), subset.size(0), dtype=torch.int, device=device)
    spd_g.index_put_((edge_index[0], edge_index[1]), edge_attr.type(torch.int))
    spd_g = spd_g + spd_g.t()

    # Mask invalid distance
    mask = ~torch.eye(spd_g.size(0), dtype=torch.bool, device=device)
    mask = mask & (spd_g <= 0)
    spd_g[mask] = INF8
    return spd_g[inv_id][:, inv_id]


def collate(idx, ids, graph, device='cpu', std=0.0):
    """
    Only support device='cpu' if num_workers > 0
    """
    idx = torch.stack(idx, dim=0).flatten()
    batch_size = idx.size(0)
    _, ns, s_total = ids.size()
    n_seq = batch_size * ns

    ids = ids[idx].view(n_seq, -1)  # [batch_size * ns, s_total]
    y = graph.y[ids[:, 0].view(-1)].view(-1)

    attn_bias = torch.empty((n_seq, s_total, s_total),
        dtype=torch.int,
        device=device,
    )
    for g, subnodes in enumerate(ids):
        attn_bias[g] = gen_attn_bias(graph, subnodes, device=device)

    x = graph.x[ids.view(-1)].view(n_seq, s_total, -1)
    if std > 0:
        x += torch.normal(0, std, x.shape, device=x.device, dtype=x.dtype)
    return Batch(attn_bias, x, y)
