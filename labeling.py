import torch
from queue import Queue
from tqdm import tqdm

INF = 1e9

def query(T, x, labels):
    ret = INF
    for u, dis in labels[x].items():
        ret = min(ret, dis + T[u])
    return ret


def bfs(src, labels, neighbors, dis, T):
    q = Queue()
    q.put(src)
    dis[src] = T[src] = 0
    for u, d in labels[src].items():
        T[u] = d

    while not q.empty():
        x = q.get()
        if query(T, x, labels) <= dis[x]:
            dis[x] = INF
            continue
        labels[x][src] = dis[x]
        for u in neighbors[x]:
            if dis[u] == INF:
                dis[u] = dis[x] + 1
                q.put(u)
        dis[x] = INF

    for u, _ in labels[src].items():
        T[u] = INF

def labeling(edge_index):
    print('start landmark labeing...')
    N = edge_index.max().item() + 1
    M = edge_index.shape[1]
    neighbors = [[] for _ in range(N)]
    dis = [INF for _ in range(N)]
    t = [INF for _ in range(N)]
    labels = {}
    for i in range(N):
        labels[i] = {}
    for i in range(M):
        u, v = edge_index[0][i].item(), edge_index[1][i].item()
        neighbors[u].append(v)

    K = N
    for i in tqdm(range(K)):
        bfs(i, labels, neighbors, dis, t)

    for i in range(N):
        for key, value in list(labels[i].items()):
            if value < 2:
                labels[i].pop(key)
    return labels


if __name__ == '__main__':
    name = 'cora'
    edge_index = torch.load('./dataset/'+name+'/edge_index.pt')
    labeling(edge_index)
