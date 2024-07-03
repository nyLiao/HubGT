import os
import time
import torch
import subprocess
from queue import Queue
from tqdm import tqdm

INF = 1e9

def query(T, labels):
    ret = INF
    for u, dis in labels.items():
        ret = min(ret, dis + T[u])
    return ret


def bfs(src, labels, neighbors, dis, T):
    q = Queue()
    q.put(src)
    updated = []
    dis[src] = T[src] = 0
    for u, d in labels[src].items():
        T[u] = d

    maxdis = 0
    while not q.empty():
        x = q.get()
        maxdis = max(maxdis, dis[x])
        updated.append(x)
        if query(T, labels[x]) <= dis[x]:
            continue
        labels[x][src] = dis[x]
        for u in neighbors[x]:
            if dis[u] == INF:
                dis[u] = dis[x] + 1
                q.put(u)

    # print(len(updated), maxdis)
    for x in updated:
        dis[x] = INF
    for u, _ in labels[src].items():
        T[u] = INF

def labeling(edge_index):
    print('start landmark labeing...')
    N = edge_index.max().item() + 1
    M = edge_index.shape[1]
    neighbors = [[] for _ in range(N)]
    deg = [0 for _ in range(N)]
    dis = [INF for _ in range(N)]
    t = [INF for _ in range(N)]
    labels = {}
    for i in range(N):
        labels[i] = {}
    for i in range(M):
        u, v = edge_index[0][i].item(), edge_index[1][i].item()
        neighbors[u].append(v)
        deg[u] += 1

    nodes = []
    for i in range(N):
        nodes.append((deg[i], i))
    nodes = sorted(nodes, reverse=True)

    K = 100
    for i in range(K):
        bfs(nodes[i][1], labels, neighbors, dis, t)

    for i in range(N):
        for key, value in list(labels[i].items()):
            if value < 2:
                labels[i].pop(key)
    return labels


def construct_index(name, edge_index, K):
    index_path = "./dataset/" + name + f"/index_file_{K}.txt"
    if os.path.exists(index_path):
        return 
    edge_path = "./dataset/" + name + "/edges.txt"
    f = open(edge_path, "w")
    for i in range(edge_index.shape[1]):
        f.write(f"{edge_index[0][i].item()} {edge_index[1][i].item()}\n")
    f.close()
    cmd = f"pll/bin/construct_index {edge_path} {K} 0 {index_path}"
    print('start landmark labeling...')
    start = time.time()
    subprocess.call(cmd, shell=True)
    print(f"done! time: {time.time() - start:.2f}s")


def generate_kspd(name, queries, K):
    vis = {}
    kspd_path = "./dataset/" + name + f"/kspd_{K}.txt"
    if os.path.exists(kspd_path):
        return 
    index_path = "./dataset/" + name + f"/index_file_{K}.txt"
    query_path = "./dataset/" + name + f"/queries_{K}.txt"
    query_file = open(query_path, "w")
    for u, v in queries:
        if (u, v) in vis or (v, u) in vis:
            continue
        query_file.write(f"{u} {v}\n")
        vis[(u, v)] = True
    query_file.close()

    cmd = f"pll/bin/k_distance {K} {index_path} {kspd_path} < {query_path}"
    print('start generate KSPD...')
    start = time.time()
    subprocess.call(cmd, shell=True)
    print(f"done! time: {time.time() - start:.2f}s")


def load_kspd(path):
    kspd_map = {}
    kspd_file = open(path, "r")
    lines = kspd_file.readlines()
    for line in lines:
        feat = line.strip().split(' ')
        u, v, kspd = int(feat[0]), int(feat[1]), [int(x) for x in feat[2:]]
        kspd = torch.FloatTensor(kspd)
        kspd[kspd == -1] = 128
        kspd = torch.clamp(kspd, 0, 128)
        kspd = 1 / torch.sqrt(kspd + 1)
        kspd_map[(u, v)] = kspd
    return kspd_map


def self_kspd_feature(feature, name, N, K):
    index_path = "./dataset/" + name + f"/index_file_{K}.txt"
    query_path = "./dataset/" + name + f"/queries_{K}.txt"
    kspd_path = "./dataset/" + name + f"/kspd_{K}.txt"
    query_file = open(query_path, "w")
    for i in range(N):
        query_file.write(f"{i} {i}\n")
    query_file.close()

    cmd = f"pll/bin/k_distance {K} {index_path} {kspd_path} < {query_path}"
    print('start generate KSPD...')
    start = time.time()
    subprocess.call(cmd, shell=True)
    print(f"done! time: {time.time() - start:.2f}s")

    kspd_map = load_kspd(kspd_path)
    kspd_feat = []
    for i in range(feature.shape[0]):
        if (i, i) in kspd_map:
            kspd_feat.append(kspd_map[(i, i)])
        else:
            kspd_feat.append(torch.zeros(K, ))
    kspd_feat = torch.stack(kspd_feat)
    return torch.cat([feature, kspd_feat], dim=-1)


if __name__ == '__main__':
    name = 'arxiv-year'
    edge_index = torch.load('./dataset/'+name+'/edge_index.pt')
    construct_index(name, edge_index, 16)
    quries = []
    N = edge_index.max()
    data = torch.load('./dataset/'+name+'/data.pt')
    for items in data:
        for item in items:
            u = item[1][0].item()
            for v in item[1]:
                if v < N:
                    quries.append((u, v.item()))
    generate_kspd(name, quries, 16)
