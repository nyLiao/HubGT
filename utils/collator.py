import numpy as np
import torch

INF8 = 255


class Batch(object):
    def __init__(self, attn_bias, x, y, ids=None):
        super(Batch, self).__init__()
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.ids = ids

    def to(self, device):
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.ids = self.ids.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def collate(idx, ids, graph, std=0.0):
    idx = torch.stack(idx, dim=0).flatten() # avoid slice and copy for ids
    batch_size = idx.size(0)
    _, ns, s_total = ids.size()
    kbias = graph.spd_bias.shape[1]
    n_seq = batch_size * ns

    ids = ids[idx].view(n_seq, -1)  # [batch_size * ns, s_total]
    y = graph.y[ids[:, 0].view(-1)].view(-1)

    attn_bias = np.empty((n_seq, s_total, s_total, kbias), dtype=np.int16)
    for g, subnodes in enumerate(ids):
        spd_g = graph.spd[subnodes][:, subnodes].toarray()
        spd_g = spd_g + spd_g.T
        # Mask invalid distance
        attn_bias_g = graph.spd_bias[spd_g]
        mask = ~np.eye(spd_g.shape[0], dtype=bool).reshape(spd_g.shape[0], spd_g.shape[0], 1).repeat(kbias, axis=2)
        mask = mask & (attn_bias_g <= 0)
        attn_bias_g[mask] = INF8
        attn_bias[g] = attn_bias_g

    x = graph.x[ids.view(-1)].view(n_seq, s_total, -1)
    if std > 0:
        norm = torch.norm(x)
        x += torch.normal(0, std * norm, x.shape, device=x.device, dtype=x.dtype)

    return Batch(torch.from_numpy(attn_bias), x, y, ids)


def collate_sim(idx, ids, graph, std=0.0):
    idx = torch.stack(idx, dim=0).flatten() # avoid slice and copy for ids
    batch_size = idx.size(0)
    _, ns, s_total = ids.shape
    n_seq = batch_size * ns

    ids = ids[idx].view(n_seq, -1)  # [batch_size * ns, s_total]
    y = graph.y[ids[:, 0].view(-1)].view(-1)
    attn_bias = graph.attn_bias[idx].view(n_seq, s_total, s_total, -1)

    x = graph.x[ids.view(-1)].view(n_seq, s_total, -1)
    if std > 0:
        norm = torch.norm(x)
        x += torch.normal(0, std * norm, x.shape, device=x.device, dtype=x.dtype)

    return Batch(attn_bias, x, y, ids)


def collate_fetch(idx, c_handler, graph, s_total, std=0.0):
    idx = torch.stack(idx, dim=0).flatten() # avoid slice and copy for ids
    batch_size = idx.size(0)

    ids, attn_bias = c_handler.fetch_parallel(idx.numpy().astype(np.int32), s_total)
    ids = torch.tensor(ids, dtype=int).view(batch_size, s_total)
    attn_bias = torch.tensor(attn_bias, dtype=int).view(batch_size, s_total, s_total, 1)

    y = graph.y[ids[:, 0].view(-1)].view(-1)
    x = graph.x[ids.view(-1)].view(batch_size, s_total, -1)
    if std > 0:
        norm = torch.norm(x)
        x += torch.normal(0, std * norm, x.shape, device=x.device, dtype=x.dtype)

    return Batch(attn_bias, x, y, ids)
