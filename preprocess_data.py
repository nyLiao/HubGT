import os
import time
import argparse
import numpy as np
import torch
import os.path as osp
import pickle
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
from numpy.linalg import inv
from torch_geometric.datasets import Planetoid, Amazon
from pygsp import graphs
from graph_coarsening.coarsening_utils import coarsen
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_sparse import coalesce
from tqdm import tqdm
from labeling import labeling
from datasets import load_nc_dataset


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def coarse_graph_adj(mx, p):
    p[p > 0] = 1.
    p = np.array(p)
    rowsum = p.sum(1)
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    p = np.matmul(p.T, r_mat_inv)
    mx = np.matmul(mx.toarray(), p)
    mx = np.matmul(p.T, mx)
    return mx


def coarse_adj_normalize(adj):
    adj += np.diag(np.ones(adj.shape[0]))
    r_inv = np.power(adj.sum(1), -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    adj = np.matmul(r_mat_inv, adj)
    adj = np.matmul(adj, r_mat_inv)
    return adj


def process_data(name, use_coarsen_feature=True):
    dataset = load_nc_dataset(dataname=name)
    edge_index = dataset.graph['edge_index']
    x = dataset.graph['node_feat']
    y = dataset.label.reshape(-1)

    subgraphs = labeling(edge_index)
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                shape=(y.shape[0], y.shape[0]),
                                dtype=np.float32)
    normalized_adj = adj_normalize(adj)
    column_normalized_adj = column_normalize(adj)

    if not os.path.exists('./dataset/' + name):
        os.makedirs('./dataset/' + name)
    sp.save_npz('./dataset/'+name+'/normalized_adj.npz', normalized_adj)
    sp.save_npz('./dataset/' + name + '/column_normalized_adj.npz', column_normalized_adj)
    c = 0.15
    k0 = 15
    k1 = 15
    k2 = 0
    Samples = 8 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    torch.save(x, './dataset/' + name + '/x.pt')
    torch.save(y, './dataset/' + name + '/y.pt')
    torch.save(edge_index, './dataset/' + name + '/edge_index.pt')

    # Sampling heuristics: 0,1,2,3
    eigen_adj = sp.eye(adj.shape[0])
    for _ in range(5):
        eigen_adj = c * sp.eye(adj.shape[0]) + (1-c) * normalized_adj * eigen_adj
    eigen_adj = eigen_adj.tocsr()

    if use_coarsen_feature:
        print('Coarsening Graph...')
        start = time.time()
        G = graphs.Graph(adj + adj.T)
        C, Gc, _, _ = coarsen(G, K=10, r=0.9, method='algebraic_JC')
        print(f"Done! Time: {time.time() - start:.2f}s, Number of super nodes: {C.shape[0]}")

        C = torch.tensor(C/C.sum(1), dtype=torch.float32)
        super_node_feature = torch.matmul(C, x)
        feature = torch.cat([x, super_node_feature])
        node_supernode_dict = {}
        for i in range(y.shape[0]):
            node_supernode_dict[i] = torch.where(C[:, i] > 0)[0].item()
        Coarse_adj = coarse_graph_adj(adj, C)
        Coarse_graph_dim = Coarse_adj.shape[0]
        normalized_coarse_graph = coarse_adj_normalize(Coarse_adj)
        coarse_power_adj_list = [normalized_coarse_graph]
        for m in range(5):
            coarse_power_adj_list.append(np.matmul(normalized_coarse_graph, coarse_power_adj_list[m]))
    else:
        feature = x

    # Create subgraph samples
    print('creating subgraph samples...')
    data_list = []
    for id in tqdm(range(y.shape[0])):
        sub_data_list = []
        s = eigen_adj[id].toarray()[0]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-(k0+k1+k2):]
        if use_coarsen_feature:
            super_node_id = node_supernode_dict[id]

        s = eigen_adj[id].toarray()[0]
        s[id] = 0
        s = np.maximum(s, 0)

        sample_num0 = np.minimum(k0, len(subgraphs[id]))
        sample_num1 = np.minimum(k1 + (k0 - sample_num0), (s > 0).sum())
        sample_num2 = np.minimum(k2, (Coarse_adj[super_node_id] > 0).sum()) if use_coarsen_feature else 0
        #create subgraph samples for ensemble
        for _ in range(Samples):
            if sample_num0 > 0:
                sample_index0 = np.random.choice(a=np.arange(len(subgraphs[id])), size=sample_num0, replace=False)
                subgraph_nodes = np.array(list(subgraphs[id].keys()), dtype=int)
                sample_index0 = subgraph_nodes[sample_index0]
            else:
                sample_index0 = np.array([], dtype=int)
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)
            if sample_num2 > 0:
                sample_index2 = np.random.choice(a=np.arange(Coarse_graph_dim), size=sample_num2, replace=False, p=Coarse_adj[super_node_id]/Coarse_adj[super_node_id].sum())
            else:
                sample_index2 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index0, dtype=int), torch.tensor(sample_index1, dtype=int),
                                    torch.tensor(top_neighbor_index[: k0+k1+k2-sample_num2-sample_num1-sample_num0], dtype=int)])
            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])

            if use_coarsen_feature:
                super_node_list = np.concatenate([[super_node_id], sample_index2])
                node2supernode_list = np.array([node_supernode_dict[i.item()] for i in node_feature_id])
                all_node_list = np.concatenate([node2supernode_list, super_node_list])

                attn_bias_complem1 = torch.cat([torch.tensor(i[super_node_list, :][:, node2supernode_list], dtype=torch.float32).unsqueeze(0) for i in coarse_power_adj_list])
                attn_bias_complem2 = torch.cat([torch.tensor(i[all_node_list, :][:, super_node_list], dtype=torch.float32).unsqueeze(0) for i in coarse_power_adj_list])

                attn_bias = torch.cat([attn_bias, attn_bias_complem1], dim=1)
                attn_bias = torch.cat([attn_bias, attn_bias_complem2], dim=2)

                label = torch.cat([y[node_feature_id], torch.zeros(len(super_node_list))]).long()
                feature_id = torch.cat([node_feature_id, torch.tensor(super_node_list + y.shape[0], dtype=int)])
                assert len(feature_id) == k0+k1+k2+2
            else:
                label = y[node_feature_id]
                feature_id = node_feature_id
                assert len(feature_id) == k0+k1+k2+1

            attn_bias = attn_bias.permute(1, 2, 0)
            sub_data_list.append([attn_bias, feature_id, label])

        data_list.append(sub_data_list)

    torch.save(data_list, './dataset/'+name+'/data.pt')
    torch.save(feature, './dataset/'+name+'/feature.pt')
    print('done!')

    return data_list, feature, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='Cora')
    args = parser.parse_args()
    process_data(args.name)



