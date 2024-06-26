import torch
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
    for i in tqdm(range(K)):
        bfs(nodes[i][1], labels, neighbors, dis, t)

    for i in range(N):
        for key, value in list(labels[i].items()):
            if value < 2:
                labels[i].pop(key)
    return labels


if __name__ == '__main__':
    name = 'ogbn_arxiv'
    edge_index = torch.load('./dataset/'+name+'/edge_index.pt')
    labeling(edge_index)
