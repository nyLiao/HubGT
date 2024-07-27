import torch
import math
import torch.nn as nn
from torch.nn import functional as F

INF8 = 255


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class StructuralEmbedding(nn.Module):
    def __init__(self, num_heads, num_global_node):
        super(StructuralEmbedding, self).__init__()
        self.num_global_node = num_global_node
        self.linear_bias = nn.Embedding(INF8+1, num_heads, padding_idx=INF8)
        if self.num_global_node > 0:
            self.virtual_bias = nn.Embedding(self.num_global_node, num_heads)

    def forward(self, attn_bias):
        # assert attn_bias.size(3) == 1     # attn_bias_dim == 1
        attn_bias = attn_bias.squeeze(3)
        n_graph, n_node = attn_bias.size()[:2]

        mask_off = (attn_bias == INF8)                      # [b, s_total, s_total]
        attn_bias = self.linear_bias(attn_bias.int())       # [b, s_total, s_total, h]
        attn_bias[mask_off] = -torch.inf

        # Append virtual node
        if self.num_global_node > 0:
            vnode_attn_bias = self.virtual_bias.weight.unsqueeze(0)
            attn_bias = torch.cat([                         # [b, s_total+nv, s_total, h]
                attn_bias,
                vnode_attn_bias.unsqueeze(2).repeat(n_graph, 1, n_node, 1)], dim=1)
            attn_bias = torch.cat([                         # [b, s_total+nv, s_total+nv, h]
                attn_bias,
                vnode_attn_bias.unsqueeze(0).repeat(n_graph, n_node+self.num_global_node, 1, 1)], dim=2)

        return attn_bias.permute(0, 3, 1, 2)                # [b, h, s_total+nv, s_total+nv]


class StructuralLinear(nn.Module):
    def __init__(self, num_heads, attn_bias_dim, num_global_node):
        super(StructuralLinear, self).__init__()
        self.num_global_node = num_global_node
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        if self.num_global_node > 0:
            self.virtual_bias = nn.Embedding(self.num_global_node, attn_bias_dim)

    def forward(self, attn_bias):
        # Append virtual node
        if self.num_global_node > 0:
            n_graph, n_node = attn_bias.size()[:2]
            vnode_attn_bias = self.virtual_bias.weight.unsqueeze(0)
            attn_bias = torch.cat([
                attn_bias,
                vnode_attn_bias.unsqueeze(2).repeat(n_graph, 1, n_node, 1)], dim=1)
            attn_bias = torch.cat([
                attn_bias,
                vnode_attn_bias.unsqueeze(0).repeat(n_graph, n_node+self.num_global_node, 1, 1)], dim=2)

        attn_bias = self.linear_bias(attn_bias.float())     # [b, s_total, s_total, h]
        return attn_bias.permute(0, 3, 1, 2)                # [b, h, s_total, s_total]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim, num_global_node):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.bias_enc = StructuralEmbedding(num_heads, num_global_node)
        # self.bias_enc = StructuralLinear(num_heads, attn_bias_dim, num_global_node)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, get_score=False):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = self.bias_enc(attn_bias)
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        if get_score:
            score = x[:, :, 0, :] * torch.norm(v, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        if get_score:
            return x, score.mean(dim=1)
        else:
            return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim, num_global_node):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim, num_global_node)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None, get_score=False):
        y = self.self_attention_norm(x)
        if get_score:
            _, score = self.self_attention(y, y, y, attn_bias, get_score=True)
            return score
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class MyEncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MyEncoderLayer, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads

        self.beta = torch.nn.Parameter(torch.ones(num_heads * att_size)*0.1)
        self.lns = torch.nn.LayerNorm(num_heads * att_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, x, attn_bias=None, get_score=False):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        h = self.linear_h(x)
        q = F.sigmoid(self.linear_q(x)).view(batch_size, -1, d_k, self.num_heads)
        k = F.sigmoid(self.linear_k(x)).view(batch_size, -1, d_k, self.num_heads)
        v = self.linear_v(x).view(batch_size, -1, d_v, self.num_heads)

        # numerator
        kv = torch.einsum('bndh, bnmh -> bdmh', k, v)
        num = torch.einsum('bndh, bdmh -> bnmh', q, kv)

        # denominator
        k_sum = torch.einsum('bndh -> bdh', k)
        den = torch.einsum('bndh, bdh -> bnh', q, k_sum).unsqueeze(2)

        # linear global attention based on kernel trick
        x = (num/den).reshape(batch_size, -1, self.num_heads * d_v)
        x = self.lns(x) * (h + self.beta)
        x = F.relu(self.output_layer(x))
        x = self.att_dropout(x)
        return x


class GT(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        num_global_node,
        attention_dropout_rate,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_global_node = num_global_node
        if self.num_global_node > 0:
            self.virtual_feat = nn.Embedding(self.num_global_node, hidden_dim)

        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim, num_global_node)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.attn_ego = nn.Linear(hidden_dim*2, 1)
        # self.downstream_out_proj = nn.Linear(hidden_dim*8, output_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, x = batched_data.attn_bias, batched_data.x
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()         # [n_graph, n_node, n_node, hop]
        node_feature = self.node_encoder(x)         # [n_graph, n_node, n_hidden]

        if self.num_global_node > 0:
            vnode_feature = self.virtual_feat.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            node_feature = torch.cat([node_feature, vnode_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)              # [n_graph, n_node, n_hidden]

        # output part
        target = output[:, 0, :].unsqueeze(1).repeat(1, n_node-1, 1)
        out_ego, out_neighbor = torch.split(output, [1, n_node-1], dim=1)
        alpha_ego = self.attn_ego(torch.cat([target, out_neighbor], dim=2))
        alpha_ego = torch.softmax(alpha_ego, dim=1)
        out_neighbor = torch.sum(out_neighbor * alpha_ego, dim=1, keepdim=True)
        output = (out_ego + out_neighbor).squeeze(1)
        output = self.downstream_out_proj(output)
        # output = self.downstream_out_proj(output[:, 0, :])
        return output
