import numpy as np
import torch

INF8 = 255


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


def collate(idx, ids, graph, std=0.0):
    idx = torch.stack(idx, dim=0).flatten()
    batch_size = idx.size(0)
    _, ns, s_total = ids.size()
    kbias = graph.spd_bias.size(1)
    n_seq = batch_size * ns

    ids = ids[idx].view(n_seq, -1)  # [batch_size * ns, s_total]
    y = graph.y[ids[:, 0].view(-1)].view(-1)

    attn_bias = torch.empty((n_seq, s_total, s_total, kbias), dtype=torch.int16)
    for g, subnodes in enumerate(ids):
        spd_g = graph.spd[subnodes][:, subnodes].toarray()
        spd_g = spd_g + spd_g.T
        # Mask invalid distance
        attn_bias_g = graph.spd_bias[spd_g]
        mask = ~torch.eye(spd_g.shape[0], dtype=bool).unsqueeze(2).expand(-1, -1, kbias)
        mask = mask & (attn_bias_g <= 0)
        attn_bias_g[mask] = INF8
        attn_bias[g] = attn_bias_g

    x = graph.x[ids.view(-1)].view(n_seq, s_total, -1)
    if std > 0:
        norm = torch.norm(x)
        x += torch.normal(0, std * norm, x.shape, device=x.device, dtype=x.dtype)
    return Batch(attn_bias, x, y)
