#ifndef PRUNED_LANDMARK_LABELING_H_
#define PRUNED_LANDMARK_LABELING_H_

#include <malloc.h>
#include <climits>
#include <limits>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <thread>
#include <climits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <set>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <utility>


class PrunedLandmarkLabeling {
public:
  void ConstructGraph(const std::vector<uint32_t> &ns, const std::vector<uint32_t> &nt, const std::vector<uint32_t> &alias_inv);
  float ConstructIndex();
  int Global(const int v, std::vector<int> &pos, std::vector<int> &dist);
  int Label(const int v, std::vector<int> &pos, std::vector<int> &dist);
  int SNeighbor(const int v, const int size, std::vector<int> &pos, std::vector<int> &dist);

  inline int QueryDistanceTwo(const int v, const std::vector<int> &nw, std::vector<int> &ret);
  inline int QueryDistance(const int v, const int w);
  int QueryDistanceLoop(const std::vector<int> &ns, const std::vector<int> &nt, size_t st, size_t ed, std::vector<int> &ret);
  int QueryDistanceParallel(const std::vector<int> &ns, const std::vector<int> &nt, std::vector<int> &ret);

  bool LoadIndex(std::ifstream &ifs);
  bool LoadIndex(const char *filename);
  bool StoreIndex(std::ofstream &ofs);
  bool StoreIndex(const char *filename);

  void SetArgs(const bool quiet_) { quiet = quiet_; }
  int GetNumVertices() { return V; }
  int GetBP() { return LEN_BP; }
  double AverageLabelSize();

  PrunedLandmarkLabeling()
      : V(0), E(0), index_(NULL) {}
  virtual ~PrunedLandmarkLabeling() {
    Free();
  }

private:
  static const uint8_t INF8;  // For unreachable pairs
  static const int LEN_BP = 128;
  static const int NUMTHREAD = 16;
  static const int MAXIDX = 32;   // max label size
  static const int MAXDIST = 16;  // max search distance

  // 4 * 33 * BP + 40 * |L|
  struct index_t {
    uint8_t  bpspt_d[LEN_BP];     // Bit-parallel Shortest Path distances
    uint64_t bpspt_s[LEN_BP][2];  // [0]: S^{-1}, [1]: S^{0}
    size_t   len_spt, len_inv, len_two;
    uint32_t *spt_v;                // PLL Shortest Path nodes (only smaller ids, sorted) | Inverse nodes (only larger ids, not sorted) | 2-hop nodes
    uint8_t  *spt_d;                // PLL Shortest Path distances
  } __attribute__((aligned(64)));   // Aligned for cache lines

  size_t V, E;
  bool quiet = false;
  index_t *index_;
  std::vector<std::vector<uint32_t> > adj;
  std::vector<uint32_t> alias, alias_inv;

  inline void Init();
  void Free();

  double GetCurrentTimeSec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

};

const uint8_t PrunedLandmarkLabeling::INF8 = std::numeric_limits<uint8_t>::max() / 2;

// ====================
void PrunedLandmarkLabeling::
ConstructGraph(const std::vector<uint32_t> &ns, const std::vector<uint32_t> &nt, const std::vector<uint32_t> &alias_inv_) {
  // Prepare the adjacency list and index space
  Free();
  this->V = 0;
  this->E = ns.size();
  V = *std::max_element(alias_inv_.begin(), alias_inv_.end()) + 1;

  // Order vertices by decreasing order of degree
  adj.resize(V);
  alias.resize(V+LEN_BP);
  alias_inv = alias_inv_;
  for (size_t i = 0; i < V; i++) alias[alias_inv[i]] = i;

  for (size_t i = 0; i < E; i++){
    adj[alias[ns[i]]].push_back(alias[nt[i]]);
  }
}

float PrunedLandmarkLabeling::
ConstructIndex() {
  double time_neighbor, time_search, time_two;
  if (!quiet) std::cout << "Building index -- Nodes: " << V << ", Edges: " << E << std::endl;

  // Bit-parallel labeling
  Init();
  time_neighbor = -GetCurrentTimeSec();
  std::vector<bool> usd(V, false);  // Used as root? (in new label)
  {
    std::vector<uint32_t> que(V);
    std::vector<uint8_t> tmp_d(V);
    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(V);
    std::vector<std::pair<uint32_t, uint32_t> > sibling_es(E);
    std::vector<std::pair<uint32_t, uint32_t> > child_es(E);

    uint32_t r = 0;
    for (int i_bpspt = 0; i_bpspt < LEN_BP; ++i_bpspt) {
      while (r < V && usd[r]) ++r;
      if (r == V) {
        for (size_t v = 0; v < V; ++v) index_[v].bpspt_d[i_bpspt] = INF8;
        continue;
      }
      usd[r] = true;
      alias[V+i_bpspt] = r;

      fill(tmp_d.begin(), tmp_d.end(), INF8);
      fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

      int que_t0 = 0, que_t1 = 0, que_h = 0;
      que[que_h++] = r;
      tmp_d[r] = 0;
      que_t1 = que_h;

      int nns = 0;
      std::sort(adj[r].begin(), adj[r].end(), std::less<uint32_t>());
      for (size_t i = 0; i < adj[r].size(); ++i) {
        const uint32_t v = adj[r][i];
        if (!usd[v]) {
          usd[v] = true;
          que[que_h++] = v;
          tmp_d[v] = 1;
          tmp_s[v].first = 1ULL << nns;
          if (++nns == MAXIDX) break;
        }
      }

      for (uint8_t d = 0; que_t0 < que_h && d < MAXDIST; ++d) {
        size_t num_sibling_es = 0, num_child_es = 0;

        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          const uint32_t v = que[que_i];

          for (size_t i = 0; i < adj[v].size(); ++i) {
            const uint32_t tv = adj[v][i];
            const uint8_t  td = d + 1;

            if (d > tmp_d[tv]);
            else if (d == tmp_d[tv]) {
              if (v < tv) {
                sibling_es[num_sibling_es].first  = v;
                sibling_es[num_sibling_es].second = tv;
                ++num_sibling_es;
              }
            } else {
              if (tmp_d[tv] == INF8) {
                que[que_h++] = tv;
                tmp_d[tv] = td;
              }
              child_es[num_child_es].first  = v;
              child_es[num_child_es].second = tv;
              ++num_child_es;
            }
          }
        }

        for (size_t i = 0; i < num_sibling_es; ++i) {
          const uint32_t v = sibling_es[i].first, w = sibling_es[i].second;
          tmp_s[v].second |= tmp_s[w].first;
          tmp_s[w].second |= tmp_s[v].first;
        }
        for (size_t i = 0; i < num_child_es; ++i) {
          const uint32_t v = child_es[i].first, c = child_es[i].second;
          tmp_s[c].first  |= tmp_s[v].first;
          tmp_s[c].second |= tmp_s[v].second;
        }

        que_t0 = que_t1;
        que_t1 = que_h;
      }

      for (size_t v = 0; v < V; ++v) {
        index_[v].bpspt_d[i_bpspt] = tmp_d[v];
        index_[v].bpspt_s[i_bpspt][0] = tmp_s[v].first;
        index_[v].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
      }
      if (!quiet && i_bpspt % (LEN_BP / 10) == 0){
        std::cout << time_neighbor+GetCurrentTimeSec() << " (" << (100 * i_bpspt / LEN_BP) << "%) " << std::flush;
      }
    }
  }
  time_neighbor += GetCurrentTimeSec();
  if (!quiet) std::cout << "| Neighbor time: " << time_neighbor << ", BPRoot Size: " << LEN_BP << std::endl;

  // Pruned labeling
  {
    // Sentinel (V, INF8) is added to all the vertices
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint8_t> > >
        tmp_idx(V, make_pair(std::vector<uint32_t>(1, V),
                             std::vector<uint8_t>(1, INF8)));
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint8_t> > >
        tmp_inv(V, make_pair(std::vector<uint32_t>(),
                             std::vector<uint8_t>()));

    std::vector<bool> vis(V);
    std::vector<uint32_t> que(V);
    std::vector<uint32_t> que_two(V);
    std::vector<uint8_t> dst_r(V + 1, INF8);
    std::vector<size_t> tmp_len_inv(V);

    time_search = -GetCurrentTimeSec();
    for (size_t r = 0; r < V; ++r) {
      if (usd[r]) continue;
      const index_t &idx_r = index_[r];
      const std::pair<std::vector<uint32_t>, std::vector<uint8_t> >
          &tmp_idx_r = tmp_idx[r];
      std::pair<std::vector<uint32_t>, std::vector<uint8_t> >
          &tmp_inv_r = tmp_inv[r];
      for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
        dst_r[tmp_idx_r.first[i]] = tmp_idx_r.second[i];
      }

      int que_t0 = 0, que_t1 = 1, que_h = 1;
      que[0] = r;
      vis[r] = true;

      for (uint8_t d = 0; que_t0 < que_h && d < MAXDIST; ++d) {
        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          const uint32_t v = que[que_i];
          std::pair<std::vector<uint32_t>, std::vector<uint8_t> >
              &tmp_idx_v = tmp_idx[v];
          const index_t &idx_v = index_[v];

          // Prefetch
          _mm_prefetch(&idx_v.bpspt_d[0], _MM_HINT_T0);
          _mm_prefetch(&idx_v.bpspt_s[0][0], _MM_HINT_T0);
          _mm_prefetch(&tmp_idx_v.first[0], _MM_HINT_T0);
          _mm_prefetch(&tmp_idx_v.second[0], _MM_HINT_T0);

          // Prune?
          if (usd[v]) continue;
          for (int i = 0; i < LEN_BP; ++i) {
            uint8_t td = idx_r.bpspt_d[i] + idx_v.bpspt_d[i];
            if (td - 2 <= d) {
              td +=
                  (idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
                  ((idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
                   (idx_r.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                  ? -1 : 0;
              if (td <= d) goto pruned;
            }
          }
          for (size_t i = 0; i < tmp_idx_v.first.size(); ++i) {
            const uint32_t w = tmp_idx_v.first[i];
            const uint8_t td = tmp_idx_v.second[i] + dst_r[w];
            if (td <= d) goto pruned;
          }

          // Traverse
          tmp_idx_v.first .back() = r;
          tmp_idx_v.second.back() = d;
          tmp_idx_v.first .push_back(V);
          tmp_idx_v.second.push_back(INF8);
          if (d > 0) {
            tmp_inv_r.first .push_back(v);
            tmp_inv_r.second.push_back(d);
          }
          for (size_t i = 0; i < adj[v].size(); ++i) {
            const uint32_t w = adj[v][i];
            if (!vis[w] && tmp_idx[w].first.size() < MAXIDX) {
              vis[w] = true;
              if (usd[w]) continue;
              que[que_h++] = w;
            }
          }
        pruned:
          {}
        }

        que_t0 = que_t1;
        que_t1 = que_h;
      }

      for (int i = 0; i < que_h; ++i) vis[que[i]] = false;
      for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
        dst_r[tmp_idx_r.first[i]] = INF8;
      }
      usd[r] = true;
      tmp_len_inv[r] = tmp_inv_r.first.size();

      if (!quiet && r % (V / 20) == 0){
        std::cout << time_search+GetCurrentTimeSec() << " (" << (100 * r / V) << "%) " << std::flush;
      }
    }
    time_search += GetCurrentTimeSec();

    time_two = -GetCurrentTimeSec();
    for (size_t r = 0; r < V; ++r) {
      const std::pair<std::vector<uint32_t>, std::vector<uint8_t> >
          &tmp_idx_r = tmp_idx[r];
      std::pair<std::vector<uint32_t>, std::vector<uint8_t> >
          &tmp_inv_r = tmp_inv[r];

      // 2-hop neighbors
      int que_two_h = -1;

      for (size_t i = 0; ; ++i) {
        const uint32_t v = tmp_idx_r.first[i];
        if (v >= r) break;
        const uint8_t d_rv = tmp_idx_r.second[i];
        const std::pair<std::vector<uint32_t>, std::vector<uint8_t> >
            &tmp_inv_v = tmp_inv[v];

        _mm_prefetch(&tmp_inv_v.first[0], _MM_HINT_T0);
        _mm_prefetch(&tmp_inv_v.second[0], _MM_HINT_T0);

        for (size_t j = 0; j < tmp_len_inv[v]; ++j) {
          const uint32_t w = tmp_inv_v.first[j];
          if (w >= r) continue;

          const uint8_t td = d_rv + tmp_inv_v.second[j];
          if (td < dst_r[w]) {
            dst_r[w] = td;
            que_two[++que_two_h] = w;
          }
        }
      }

      for ( ; que_two_h >= 0; --que_two_h) {
        const uint32_t w = que_two[que_two_h];
        if (dst_r[w] == INF8) continue;
        tmp_inv_r.first .push_back(w);
        tmp_inv_r.second.push_back(dst_r[w]);
        dst_r[w] = INF8;
      }
    }
    time_two += GetCurrentTimeSec();

    for (size_t v = 0; v < V; ++v) {
      const size_t sz_idx = tmp_idx[v].first.size();
      const size_t sz_inv = tmp_inv[v].first.size();
      index_[v].len_spt = sz_idx;
      index_[v].len_inv = tmp_len_inv[v];
      index_[v].len_two = sz_inv - tmp_len_inv[v];

      const size_t sz_all = sz_idx + sz_inv;
      index_[v].spt_v = (uint32_t*)memalign(64, sz_all * sizeof(uint32_t));
      index_[v].spt_d = (uint8_t *)memalign(64, sz_all * sizeof(uint8_t ));
      for (size_t i = 0; i < sz_idx; ++i) index_[v].spt_v[i] = tmp_idx[v].first[i];
      for (size_t i = 0; i < sz_idx; ++i) index_[v].spt_d[i] = tmp_idx[v].second[i];
      for (size_t i = 0; i < sz_inv; ++i)  index_[v].spt_v[sz_idx+i] = tmp_inv[v].first[i];
      for (size_t i = 0; i < sz_inv; ++i)  index_[v].spt_d[sz_idx+i] = tmp_inv[v].second[i];

      tmp_idx[v].first.clear();
      tmp_idx[v].second.clear();
      tmp_inv[v].first.clear();
      tmp_inv[v].second.clear();
    }
  }

  const std::vector<int> tmp_nt = {0, 629, 632, 639, 692};
  for (size_t v = 0; v < V; ++v) {
    const size_t sz_idx = index_[v].len_spt;
    const size_t sz_inv = index_[v].len_inv + index_[v].len_two;
    const size_t sz_all = sz_idx + sz_inv;

    // if (v % 1000 == 0) {
    if (index_[v].len_two > 3) {
      std::cout << v << "(" << index_[v].len_spt << "," << index_[v].len_inv << "," << index_[v].len_two << "): ";
      for (size_t i = sz_idx; i < sz_idx+index_[v].len_inv; ++i) {
        std::cout << index_[v].spt_v[i] << "/" << (int)index_[v].spt_d[i] << " ";
      }
      std::cout << "; ";
      for (size_t i = sz_idx+index_[v].len_inv; i < sz_all; ++i) {
        std::cout << index_[v].spt_v[i] << "/" << (int)index_[v].spt_d[i] << " ";
      }
      std::cout << std::endl;

      std::vector<int> ret(5);
      QueryDistanceTwo(v, tmp_nt, ret);
      for (size_t i = 0; i < tmp_nt.size(); ++i) {
        std::cout << ret[i] << " ";
      }
      std::cout << std::endl;
      for (size_t i = 0; i < tmp_nt.size(); ++i) {
        std::cout << QueryDistance(v, tmp_nt[i]) << " ";
      }
      std::cout << std::endl;
    }
  }

  if (!quiet) std::cout << "| Search time: " << time_search << ", Two-hop time: " << time_two
    << ", Avg Label Size: " << AverageLabelSize() << std::endl;
  return time_neighbor + time_search + time_two;
}

// ====================
int PrunedLandmarkLabeling::
Global(const int v, std::vector<int> &pos, std::vector<int> &dist){
  pos.clear();
  dist.clear();
  const index_t &idx_v = index_[alias[v]];
  uint8_t d = INF8;

  for (int i = 0; i < LEN_BP; ++i){
    if (idx_v.bpspt_d[i] > MAXDIST/4) continue;
    uint8_t td = idx_v.bpspt_d[i];
    if (td <= d) {
      d = td;
      if ((d == 0) && (alias[V+i] == v)) break;
      if (d == 0) d++;
      pos.push_back(alias_inv[alias[V+i]]);
      dist.push_back(d);
    }
  }
  return pos.size();
}

int PrunedLandmarkLabeling::
Label(const int v, std::vector<int> &pos, std::vector<int> &dist){
  pos.clear();
  dist.clear();
  const index_t &idx_v = index_[alias[v]];

  for (size_t i = 0; idx_v.spt_v[i] != V; ++i){
    if (idx_v.spt_d[i] == 0) continue;
    pos.push_back(alias_inv[idx_v.spt_v[i]]);
    dist.push_back(idx_v.spt_d[i]);
  }
  return pos.size();
}

int PrunedLandmarkLabeling::
SNeighbor(const int v, const int size, std::vector<int> &pos, std::vector<int> &dist){
  pos.clear();
  dist.clear();

  std::queue<uint32_t> node_que, dist_que;
  std::vector<bool> updated(V, false);
  int d_last = 0;
  node_que.push(alias[v]);
  dist_que.push(0);
  updated[alias[v]] = true;

  while (!node_que.empty() && pos.size() < size_t(4*size*size)){
    uint32_t u = node_que.front();
    uint8_t  d = dist_que.front();
    node_que.pop();
    dist_que.pop();
    // Exit condition for every hop
    if (d > d_last && pos.size() >= size_t(size)) break;
    d_last = d;

    if (adj[u].size() > MAXIDX){
      size_t i;
      for (size_t ii = 0; i < MAXIDX; ii++){
        i = rand() % adj[u].size();
        if (updated[adj[u][i]]) continue;
        node_que.push(adj[u][i]);
        dist_que.push(d + 1);
        pos.push_back(alias_inv[adj[u][i]]);
        dist.push_back(d + 1);
        updated[adj[u][i]] = true;
      }
    }
    else{
      for (size_t i = 0; i < adj[u].size(); i++){
        if (updated[adj[u][i]]) continue;
        node_que.push(adj[u][i]);
        dist_que.push(d + 1);
        pos.push_back(alias_inv[adj[u][i]]);
        dist.push_back(d + 1);
        updated[adj[u][i]] = true;
      }
    }
  }

  return pos.size();
}

// ====================
int PrunedLandmarkLabeling::
QueryDistanceTwo(const int v, const std::vector<int> &nw, std::vector<int> &ret) {
  // TODO: unordered_map
  const index_t &idx_v = index_[v];
  const size_t len_v = idx_v.len_spt + idx_v.len_inv + idx_v.len_two;
  const uint64_t *bps_v = idx_v.bpspt_s[0];
  const uint8_t  *bpd_v = idx_v.bpspt_d;
  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);
  _mm_prefetch(&bps_v[0], _MM_HINT_T0);
  _mm_prefetch(&bpd_v[0], _MM_HINT_T0);

  for (size_t j = 0; j < nw.size(); ++j) {
    if (nw[j] >= v) {
      ret[j] = 0;
      continue;
    }

    for (size_t i = 0; ; ++i) {
      if (i == len_v) {
        const index_t &idx_w = index_[nw[j]];
        uint8_t d = INF8;
        for (int i = 0; i < LEN_BP; ++i) {
          uint8_t td = idx_v.bpspt_d[i] + idx_w.bpspt_d[i];
          if (td - 2 <= d) {
            td +=
                (idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][0]) ? -2 :
                ((idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][1]) | (idx_v.bpspt_s[i][1] & idx_w.bpspt_s[i][0]))
                ? -1 : 0;

            if (td < d) d = td;
          }
        }

        ret[j] = d;
        break;
      }
      if (idx_v.spt_v[i] == nw[j]) {
        ret[j] = idx_v.spt_d[i];
        break;
      }
    }
  }

  return len_v;
}

int PrunedLandmarkLabeling::
QueryDistance(const int v, const int w) {
  if (v >= V+LEN_BP || w >= V+LEN_BP) return v == w ? 0 : INT_MAX;

  // const index_t &idx_v = index_[alias[v]];
  // const index_t &idx_w = index_[alias[w]];
  const index_t &idx_v = index_[v];
  const index_t &idx_w = index_[w];
  uint8_t d = INF8;

  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_d[0], _MM_HINT_T0);

  for (int i = 0; i < LEN_BP; ++i) {
    uint8_t td = idx_v.bpspt_d[i] + idx_w.bpspt_d[i];
    if (td - 2 <= d) {
      td +=
          (idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][0]) ? -2 :
          ((idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][1]) | (idx_v.bpspt_s[i][1] & idx_w.bpspt_s[i][0]))
          ? -1 : 0;

      if (td < d) d = td;
    }
  }

  for (uint32_t i1 = 0, i2 = 0; ; ) {
    uint32_t v1 = idx_v.spt_v[i1], v2 = idx_w.spt_v[i2];
    if (v1 == v2) {
      if (v1 == V) break;  // Sentinel
      uint8_t td = idx_v.spt_d[i1] + idx_w.spt_d[i2];
      if (td < d) d = td;
      ++i1;
      ++i2;
    } else {
      if (v1 < v2) ++i1;
      else ++i2;
    }
  }

  if (d >= INF8 - 2) d = 127;
  return d;
}

int PrunedLandmarkLabeling::
QueryDistanceLoop(const std::vector<int> &ns, const std::vector<int> &nt, size_t st, size_t ed, std::vector<int> &ret){
  for (size_t i = st; i < ed; i++){
    auto it = ret.begin() + i;
    *it = QueryDistance(ns[i], nt[i]);
    if (!quiet && st == 0 && i % (ed / 10) == 0){
      std::cout << i << " (" << (100 * i / ed) << "%) " << std::flush;
    }
  }
  return (ed - st);
}

int PrunedLandmarkLabeling::
QueryDistanceParallel(const std::vector<int> &ns, const std::vector<int> &nt, std::vector<int> &ret){
  std::vector<std::thread> threads;
  size_t st, ed = 0;
  size_t it;

  for (it = 1; it <= ns.size() % NUMTHREAD; it++) {
    st = ed;
    ed += std::ceil((float)ns.size() / NUMTHREAD);
    threads.push_back(std::thread(&PrunedLandmarkLabeling::QueryDistanceLoop, this, ref(ns), ref(nt), st, ed, ref(ret)));
  }
  for (; it <= NUMTHREAD; it++) {
    st = ed;
    ed += ns.size() / NUMTHREAD;
    threads.push_back(std::thread(&PrunedLandmarkLabeling::QueryDistanceLoop, this, ref(ns), ref(nt), st, ed, ref(ret)));
  }
  for (size_t t = 0; t < NUMTHREAD; t++)
    threads[t].join();
  std::vector<std::thread>().swap(threads);
  return ret.size();
}


// ====================
template<typename T> inline
void write_vector(std::ofstream& ofs, const std::vector<T>& data){
	const size_t count = data.size();
	ofs.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
	ofs.write(reinterpret_cast<const char*>(&data[0]), count * sizeof(T));
}

template<typename T> inline
void read_vector(std::ifstream& ifs, std::vector<T>& data){
	size_t count;
	ifs.read(reinterpret_cast<char*>(&count), sizeof(size_t));
	data.resize(count);
	ifs.read(reinterpret_cast<char*>(&data[0]), count * sizeof(T));
}

bool PrunedLandmarkLabeling::
StoreIndex(const char *filename) {
  if (!quiet) std::cout << "Saving index -- " << filename << std::endl;
  std::ofstream ofs(filename);
  return ofs && StoreIndex(ofs);
}

bool PrunedLandmarkLabeling::
StoreIndex(std::ofstream &ofs) {
#define WRITE_BINARY(value) (ofs.write((const char*)&(value), sizeof(value)))
  WRITE_BINARY(V);
  WRITE_BINARY(E);
  write_vector(ofs, alias);
  write_vector(ofs, alias_inv);
  for (size_t v = 0; v < V; ++v){
    write_vector(ofs, adj[v]);
  }

  for (size_t v = 0; v < V; ++v) {
    const index_t &idx = index_[v];
    for (int i = 0; i < LEN_BP; ++i) {
      WRITE_BINARY(idx.bpspt_d[i]);
      WRITE_BINARY(idx.bpspt_s[i][0]);
      WRITE_BINARY(idx.bpspt_s[i][1]);
    }

    WRITE_BINARY(idx.len_spt);
    WRITE_BINARY(idx.len_inv);
    WRITE_BINARY(idx.len_two);
    const size_t acc_two = idx.len_spt + idx.len_inv + idx.len_two;
    for (uint32_t i = 0; i < acc_two; ++i) {
      WRITE_BINARY(idx.spt_v[i]);
      WRITE_BINARY(idx.spt_d[i]);
    }
  }

  return ofs.good();
}

bool PrunedLandmarkLabeling::
LoadIndex(const char *filename) {
  if (!quiet) std::cout << "Loading index -- " << filename << std::endl;
  std::ifstream ifs(filename);
  return ifs && LoadIndex(ifs);
}

bool PrunedLandmarkLabeling::
LoadIndex(std::ifstream &ifs) {
#define READ_BINARY(value) (ifs.read((char*)&(value), sizeof(value)))
  Free();
  READ_BINARY(V);
  READ_BINARY(E);
  read_vector(ifs, alias);
  read_vector(ifs, alias_inv);
  adj.resize(V);
  for (size_t v = 0; v < V; ++v){
    read_vector(ifs, adj[v]);
  }

  Init();
  for (size_t v = 0; v < V; ++v) {
    index_t &idx = index_[v];
    for (int i = 0; i < LEN_BP; ++i) {
      READ_BINARY(idx.bpspt_d[i]);
      READ_BINARY(idx.bpspt_s[i][0]);
      READ_BINARY(idx.bpspt_s[i][1]);
    }

    READ_BINARY(idx.len_spt);
    READ_BINARY(idx.len_inv);
    READ_BINARY(idx.len_two);
    const size_t acc_two = idx.len_spt + idx.len_inv + idx.len_two;
    idx.spt_v = (uint32_t*)memalign(64, acc_two * sizeof(uint32_t));
    idx.spt_d = (uint8_t *)memalign(64, acc_two * sizeof(uint8_t ));
    for (uint32_t i = 0; i < acc_two; ++i) {
      READ_BINARY(idx.spt_v[i]);
      READ_BINARY(idx.spt_d[i]);
    }
  }

  return ifs.good();
}

void PrunedLandmarkLabeling::
Init() {
  index_ = (index_t*)memalign(64, V * sizeof(index_t));
  // if (index_ == NULL) {
  //   V = 0;
  //   return false;
  // }
  for (size_t v = 0; v < V; ++v) {
    index_[v].spt_v = NULL;
    index_[v].spt_d = NULL;
  }
}

void PrunedLandmarkLabeling::
Free() {
  alias.clear();
  alias_inv.clear();
  for (size_t v = 0; v < V; ++v) {
    adj[v].clear();
  }

  for (size_t v = 0; v < V; ++v) {
    free(index_[v].spt_v);
    free(index_[v].spt_d);
  }
  free(index_);
  index_ = NULL;
  V = 0;
  E = 0;
}

double PrunedLandmarkLabeling::
AverageLabelSize() {
  double s = 0.0;
  for (size_t v = 0; v < V; ++v) {
    s += index_[v].len_spt + index_[v].len_inv + index_[v].len_two;
  }
  return s / V;
}

#endif  // PRUNED_LANDMARK_LABELING_H_
