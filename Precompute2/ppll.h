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
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <utility>
#include <numeric>
#include <assert.h>

using std::vector;
using std::pair;
using std::queue;
using std::map;
using std::unordered_map;
using std::cout;
using std::endl;

class PrunedLandmarkLabeling {
public:
  void ConstructGraph(const vector<uint32_t> &ns, const vector<uint32_t> &nt, const vector<uint32_t> &alias_inv);
  float ConstructIndex();
  // node input named by `s` is external id, `v` is internal id
  int FetchNode(const int s, vector<int> &pos, vector<int> &dist);
  int FetchLoop(const vector<int> &ns, size_t st, size_t ed, vector<int> &pos, vector<int> &dist);
  int FetchParallel(const vector<int> &ns, vector<int> &pos, vector<int> &dist);
  inline int Global(const int v, vector<int> &pos, vector<int> &dist);
  inline int Label(const int v, vector<int> &pos, vector<int> &dist);
  inline int InvLabel(const int v, vector<int> &pos, vector<int> &dist);
  inline int SNeighbor(const int v, const int size, vector<int> &pos, vector<int> &dist);

  inline int QueryDistanceTwo(const int v, const vector<int> &nw, vector<int> &ret);
  inline int QueryDistance(const int v, const int w);
  int QueryDistanceLoop(const vector<int> &ns, const vector<int> &nt, size_t st, size_t ed, vector<int> &ret);
  int QueryDistanceParallel(const vector<int> &ns, const vector<int> &nt, vector<int> &ret);

  bool LoadIndex(std::ifstream &ifs);
  bool LoadIndex(const char *filename);
  bool StoreIndex(std::ofstream &ofs);
  bool StoreIndex(const char *filename);

  void SetArgs(const bool quiet_, const int seed,
    const int n_fetch, const int n_bp, const int n_spt, const int n_inv) {
    quiet = quiet_;
    SEED = seed;
    NUM_FETCH = n_fetch;
    NUM_BP = n_bp;
    NUM_SPT = n_spt;
    NUM_INV = n_inv;
  }
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
  static const uint8_t MAXDIST = 12;  // max search distance
  static const uint8_t HMAXDIST = 6;

  int SEED = 42;
  int NUM_FETCH = 48;
  int NUM_BP = 8;
  int NUM_SPT = 24;
  int NUM_INV = 12;

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
  vector<vector<uint32_t> > adj;
  vector<uint32_t> alias, alias_inv;

  inline void Init();
  void Free();
  inline int SampleSet(const int size, const vector<int> &set, const vector<int> &set2, vector<int> &ret, vector<int> &ret2);

  double GetCurrentTimeSec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

};

const uint8_t PrunedLandmarkLabeling::INF8 = std::numeric_limits<uint8_t>::max() / 2;

// ====================
void PrunedLandmarkLabeling::
ConstructGraph(const vector<uint32_t> &ns, const vector<uint32_t> &nt, const vector<uint32_t> &alias_inv_) {
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
    adj[alias[ns[i]]].emplace_back(alias[nt[i]]);
  }
}

float PrunedLandmarkLabeling::
ConstructIndex() {
  double time_neighbor, time_search, time_two;
  if (!quiet) cout << "Building index -- Nodes: " << V << ", Edges: " << E << endl;

  // Bit-parallel labeling
  Init();
  time_neighbor = -GetCurrentTimeSec();
  vector<bool> usd(V, false);  // Used as root? (in new label)
  {
    vector<uint32_t> que(V);
    vector<uint8_t> tmp_d(V);
    vector<pair<uint64_t, uint64_t> > tmp_s(V);
    vector<pair<uint32_t, uint32_t> > sibling_es(E);
    vector<pair<uint32_t, uint32_t> > child_es(E);

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
          if (++nns == 64) break;
        }
      }

      for (uint8_t d = 0; que_t0 < que_h && d < MAXDIST; ++d) {
        size_t num_sibling_es = 0, num_child_es = 0;

        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          const uint32_t v = que[que_i];

          for (size_t i = 0; i < adj[v].size(); ++i) {
            const uint32_t tv = adj[v][i];
            const uint8_t  td = d + 1;

            if (d == tmp_d[tv]) {
              if (v < tv) {
                sibling_es[num_sibling_es].first  = v;
                sibling_es[num_sibling_es].second = tv;
                ++num_sibling_es;
              }
            } else if (d < tmp_d[tv]) {
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
        cout << time_neighbor+GetCurrentTimeSec() << " (" << (100 * i_bpspt / LEN_BP) << "%) " << std::flush;
      }
    }
  }
  time_neighbor += GetCurrentTimeSec();
  if (!quiet) cout << "| Neighbor time: " << time_neighbor << ", BPRoot Size: " << LEN_BP << endl;

  // Pruned labeling
  {
    // Sentinel (V, INF8) is added to all the vertices
    vector< pair<vector<uint32_t>, vector<uint8_t>> >
        tmp_idx(V, make_pair(vector<uint32_t>(1, V), vector<uint8_t>(1, INF8)));
    vector< pair<vector<uint32_t>, vector<uint8_t>> >
        tmp_inv(V, make_pair(vector<uint32_t>(), vector<uint8_t>()));

    vector<bool> vis(V);
    vector<uint32_t> que(V);
    queue<uint32_t> que_two;
    vector<uint8_t> dst_r(V + 1, INF8);
    vector<size_t> tmp_len_inv(V);

    time_search = -GetCurrentTimeSec();
    for (size_t r = 0; r < V; ++r) {
      if (usd[r]) continue;
      const index_t &idx_r = index_[r];
      const pair<vector<uint32_t>, vector<uint8_t>> &tmp_idx_r = tmp_idx[r];
      pair<vector<uint32_t>, vector<uint8_t>> &tmp_inv_r = tmp_inv[r];
      for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
        dst_r[tmp_idx_r.first[i]] = tmp_idx_r.second[i];
      }

      int que_t0 = 0, que_t1 = 1, que_h = 1;
      que[0] = r;
      vis[r] = true;

      for (uint8_t d = 0; que_t0 < que_h && d < MAXDIST; ++d) {
        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          const uint32_t v = que[que_i];
          pair<vector<uint32_t>, vector<uint8_t>> &tmp_idx_v = tmp_idx[v];
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
          tmp_idx_v.first .emplace_back(V);
          tmp_idx_v.second.emplace_back(INF8);
          if (d > 0 && d < HMAXDIST) {
            tmp_inv_r.first .emplace_back(v);
            tmp_inv_r.second.emplace_back(d);
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
        cout << time_search+GetCurrentTimeSec() << " (" << (100 * r / V) << "%) " << std::flush;
      }
    }
    time_search += GetCurrentTimeSec();
    if (!quiet) cout << "| Search time: " << time_search << endl;

    time_two = -GetCurrentTimeSec();
    for (size_t r = 0; r < V; ++r) {
      const pair<vector<uint32_t>, vector<uint8_t>> &tmp_idx_r = tmp_idx[r];
      pair<vector<uint32_t>, vector<uint8_t>> &tmp_inv_r = tmp_inv[r];

      // 2-hop neighbors
      for (size_t i = 0; ; ++i) {
        const uint32_t v = tmp_idx_r.first[i];
        if (v >= r) break;
        const uint8_t d_rv = tmp_idx_r.second[i];
        if (d_rv > HMAXDIST) continue;
        const pair<vector<uint32_t>, vector<uint8_t>> &tmp_inv_v = tmp_inv[v];

        _mm_prefetch(&tmp_inv_v.first[0], _MM_HINT_T0);
        _mm_prefetch(&tmp_inv_v.second[0], _MM_HINT_T0);

        for (size_t j = 0; j < tmp_len_inv[v]; ++j) {
          const uint32_t w = tmp_inv_v.first[j];
          if (w >= r) continue;

          const uint8_t td = d_rv + tmp_inv_v.second[j];
          if (td > HMAXDIST) continue;
          if (td < dst_r[w]) {
            dst_r[w] = td;
            que_two.push(w);
          }
        }
      }

      while (!que_two.empty()) {
        uint32_t w = que_two.front();
        que_two.pop();
        if (dst_r[w] >= MAXDIST) continue;
        tmp_inv_r.first .emplace_back(w);
        tmp_inv_r.second.emplace_back(dst_r[w]);
        dst_r[w] = INF8;
      }

      if (!quiet && r % (V / 20) == 0){
        cout << time_two+GetCurrentTimeSec() << " (" << (100 * r / V) << "%) " << std::flush;
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

  if (!quiet) cout << "| Two-hop time: " << time_two << endl;
  if (!quiet) AverageLabelSize();
  return time_neighbor + time_search + time_two;
}

// ====================
int PrunedLandmarkLabeling::
FetchNode(const int s, vector<int> &pos, vector<int> &dist){
  const uint32_t v = alias[s];
  const index_t &idx_v = index_[v];
  size_t s_bp, s_spt, s_inv, s_adj, sn_adj, s_total;

  vector<int> tmp_pos(LEN_BP), tmp_dist(LEN_BP);
  vector<vector<int>> mat(NUM_FETCH, vector<int>{});
  vector<int> &mat0 = mat[0];

  // Construct sample
  pos.clear();
  pos.reserve(NUM_FETCH);
  pos.emplace_back(v);
  mat0.emplace_back(0);
  Global(v, tmp_pos, tmp_dist);
  s_bp = SampleSet(NUM_BP, tmp_pos, tmp_dist, pos, mat0);
  Label(v, tmp_pos, tmp_dist);
  s_spt = SampleSet(NUM_SPT, tmp_pos, tmp_dist, pos, mat0);
  InvLabel(v, tmp_pos, tmp_dist);
  s_inv = SampleSet(NUM_INV, tmp_pos, tmp_dist, pos, mat0);
  s_total = s_bp + s_spt + s_inv;
  sn_adj = NUM_FETCH - s_total - 1;
  SNeighbor(v, sn_adj, tmp_pos, tmp_dist);
  s_adj = SampleSet(sn_adj, tmp_pos, tmp_dist, pos, mat0);
  s_total += s_adj + 1;

  // Query pair-wise distance: pos (len s) -> dist (s * s)
  vector<int> argpos(s_total);
  std::iota(argpos.begin(), argpos.end(), 0);
  std::sort(argpos.begin(), argpos.end(), [&pos](int a, int b) {
    return pos[a] > pos[b];
  });

  tmp_pos.clear();
  tmp_pos.emplace_back(pos[argpos[s_total-1]]);
  for (size_t j = 1; j < s_total; ++j) {
    mat[j].resize(s_total);
    mat[j][0] = mat0[j];
  }

  for (int i = s_total-2; i >= 0; --i) {
    if (pos[argpos[i]] == v) {
      tmp_pos.emplace_back(pos[argpos[i]]);
      continue;
    }
    tmp_dist.clear();
    QueryDistanceTwo(pos[argpos[i]], tmp_pos, tmp_dist);
    // mat[argpos[i]][argpos[i]] = 0;
    for (int j = s_total-1; j > i; --j) {
      mat[argpos[i]][argpos[j]] = tmp_dist[s_total-1 - j];
      mat[argpos[j]][argpos[i]] = tmp_dist[s_total-1 - j];
    }
    tmp_pos.emplace_back(pos[argpos[i]]);
  }
  for (size_t i = 0; i < s_total; ++i) {
    pos[i] = alias_inv[pos[i]];
  }

  // Align to output
  // pos and mat[i]: s -> n
  while (pos.size() < NUM_FETCH) {
    const size_t copy_size = std::min(s_total, NUM_FETCH - pos.size());
    pos.insert(pos.end(), pos.begin(), pos.begin() + copy_size);
    for (size_t i = 0; i < s_total; ++i) {
      mat[i].insert(mat[i].end(), mat[i].begin(), mat[i].begin() + copy_size);
    }
  }

  // dist: s*n -> n*n
  dist.clear();
  dist.reserve(NUM_FETCH * NUM_FETCH);
  for (size_t i = 0; i < s_total; ++i) {
    dist.insert(dist.end(), std::make_move_iterator(mat[i].begin()),
                            std::make_move_iterator(mat[i].end()));
  }
  while (dist.size() < NUM_FETCH * NUM_FETCH) {
    const size_t copy_size = std::min(s_total * NUM_FETCH, NUM_FETCH * NUM_FETCH - dist.size());
    dist.insert(dist.end(), dist.begin(), dist.begin() + copy_size);
  }

  return s_total;
}

int PrunedLandmarkLabeling::
FetchLoop(const vector<int> &ns, size_t st, size_t ed, vector<int> &pos, vector<int> &dist){
  vector<int> tmp_pos, tmp_dist;
  for (size_t i = st; i < ed; i++){
    FetchNode(ns[i], tmp_pos, tmp_dist);
    std::copy(tmp_pos.begin(), tmp_pos.end(), pos.begin() + i * NUM_FETCH);
    std::copy(tmp_dist.begin(), tmp_dist.end(), dist.begin() + i * NUM_FETCH * NUM_FETCH);
  }
  return (ed - st);
}

int PrunedLandmarkLabeling::
FetchParallel(const vector<int> &ns, vector<int> &pos, vector<int> &dist){
  vector<std::thread> threads;
  size_t st, ed = 0;
  size_t it;

  for (it = 1; it <= ns.size() % NUMTHREAD; it++) {
    st = ed;
    ed += std::ceil((float)ns.size() / NUMTHREAD);
    threads.push_back(std::thread(&PrunedLandmarkLabeling::FetchLoop, this, ref(ns), st, ed, ref(pos), ref(dist)));
  }
  for (; it <= NUMTHREAD; it++) {
    st = ed;
    ed += ns.size() / NUMTHREAD;
    threads.push_back(std::thread(&PrunedLandmarkLabeling::FetchLoop, this, ref(ns), st, ed, ref(pos), ref(dist)));
  }
  for (size_t t = 0; t < NUMTHREAD; t++)
    threads[t].join();
  vector<std::thread>().swap(threads);
  return pos.size();
}

// ====================
int PrunedLandmarkLabeling::
Global(const int v, vector<int> &pos, vector<int> &dist){
  pos.clear();
  dist.clear();
  const index_t &idx_v = index_[v];
  _mm_prefetch(&idx_v.bpspt_d[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.bpspt_s[0][0], _MM_HINT_T0);
  uint8_t d = INF8;

  for (int i = 0; i < LEN_BP; ++i){
    if (idx_v.bpspt_d[i] > MAXDIST/2) continue;
    uint8_t td = idx_v.bpspt_d[i];
    if (td <= d) {
      d = td;
      if ((d == 0) && (V+i == v)) break;
      if (d == 0) d++;
      pos.emplace_back(alias[V+i]);
      dist.emplace_back(d);
    }
  }
  return pos.size();
}

int PrunedLandmarkLabeling::
Label(const int v, vector<int> &pos, vector<int> &dist){
  pos.clear();
  dist.clear();
  const index_t &idx_v = index_[v];
  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);

  for (size_t i = 0; idx_v.spt_v[i] != V; ++i){
    if (idx_v.spt_d[i] == 0) continue;
    pos.emplace_back(idx_v.spt_v[i]);
    dist.emplace_back(idx_v.spt_d[i]);
  }
  return pos.size();
}

int PrunedLandmarkLabeling::
InvLabel(const int v, vector<int> &pos, vector<int> &dist){
  pos.clear();
  dist.clear();
  const index_t &idx_v = index_[v];
  const size_t acc_spt = idx_v.len_spt, acc_inv = idx_v.len_spt + idx_v.len_inv;
  _mm_prefetch(&idx_v.spt_v[acc_spt], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[acc_spt], _MM_HINT_T0);

  for (size_t i = acc_spt; i < acc_inv; ++i){
    if (idx_v.spt_d[i] == 0) continue;
    pos.emplace_back(idx_v.spt_v[i]);
    dist.emplace_back(idx_v.spt_d[i]);
  }
  return pos.size();
}

int PrunedLandmarkLabeling::
SNeighbor(const int v, const int size, vector<int> &pos, vector<int> &dist){
  pos.clear();
  dist.clear();

  queue<uint32_t> node_que, dist_que;
  vector<bool> vis(V, false);
  int d_last = 0;
  node_que.push(v);
  dist_que.push(0);
  vis[v] = true;

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
        if (vis[adj[u][i]]) continue;
        node_que.push(adj[u][i]);
        dist_que.push(d + 1);
        pos.emplace_back(adj[u][i]);
        dist.emplace_back(d + 1);
        vis[adj[u][i]] = true;
      }
    }
    else{
      for (size_t i = 0; i < adj[u].size(); i++){
        if (vis[adj[u][i]]) continue;
        node_que.push(adj[u][i]);
        dist_que.push(d + 1);
        pos.emplace_back(adj[u][i]);
        dist.emplace_back(d + 1);
        vis[adj[u][i]] = true;
      }
    }
  }

  return pos.size();
}

int PrunedLandmarkLabeling::
SampleSet(const int size, const vector<int> &set, const vector<int> &set2, vector<int> &ret, vector<int> &ret2) {
  if (set.size() <= size) {
    ret.insert (ret.end(),  std::make_move_iterator(set.begin()),
                            std::make_move_iterator(set.end()));
    ret2.insert(ret2.end(), std::make_move_iterator(set2.begin()),
                            std::make_move_iterator(set2.end()));
    return set.size();
  }

  vector<bool> usd(set.size(), false);
  int cnt = 0;
  while (cnt < size) {
    int index = rand() % set.size();
    if (!usd[index]) {
      usd[index] = true;
      ret.emplace_back(set[index]);
      ret2.emplace_back(set2[index]);
      cnt++;
    }
  }
  return size;
}

// ====================
int PrunedLandmarkLabeling::
QueryDistanceTwo(const int v, const vector<int> &nw, vector<int> &ret) {
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
QueryDistanceLoop(const vector<int> &ns, const vector<int> &nt, size_t st, size_t ed, vector<int> &ret){
  for (size_t i = st; i < ed; i++){
    auto it = ret.begin() + i;
    *it = QueryDistance(ns[i], nt[i]);
    if (!quiet && st == 0 && i % (ed / 10) == 0){
      cout << i << " (" << (100 * i / ed) << "%) " << std::flush;
    }
  }
  return (ed - st);
}

int PrunedLandmarkLabeling::
QueryDistanceParallel(const vector<int> &ns, const vector<int> &nt, vector<int> &ret){
  vector<std::thread> threads;
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
  vector<std::thread>().swap(threads);
  return ret.size();
}


// ====================
template<typename T> inline
void write_vector(std::ofstream& ofs, const vector<T>& data){
	const size_t count = data.size();
	ofs.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
	ofs.write(reinterpret_cast<const char*>(&data[0]), count * sizeof(T));
}

template<typename T> inline
void read_vector(std::ifstream& ifs, vector<T>& data){
	size_t count;
	ifs.read(reinterpret_cast<char*>(&count), sizeof(size_t));
	data.resize(count);
	ifs.read(reinterpret_cast<char*>(&data[0]), count * sizeof(T));
}

bool PrunedLandmarkLabeling::
StoreIndex(const char *filename) {
  if (!quiet) cout << "Saving index -- " << filename << endl;
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
  if (!quiet) cout << "Loading index -- " << filename << endl;
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
  srand(SEED);
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
  double s_spt = 0.0, s_inv = 0.0, s_two = 0.0;
  double w_two = 0.0, n_two = 0.0;
  double n1 = 0.0, n2 = 0.0, n3 = 0.0, n4 = 0.0, n5 = 0.0, n6 = 0.0, n7 = 0.0;
  for (size_t v = 0; v < V; ++v) {
    s_spt += index_[v].len_spt;
    s_inv += index_[v].len_inv;
    s_two += index_[v].len_two;
    for (size_t i = index_[v].len_spt+index_[v].len_inv; i < index_[v].len_spt+index_[v].len_inv+index_[v].len_two; ++i) {
      switch (index_[v].spt_d[i]) {
        case 1: n1++; break;
        case 2: n2++; break;
        case 3: n3++; break;
        case 4: n4++; break;
        case 5: n5++; break;
        case 6: n6++; break;
        default: n7++; break;
      }
      n_two++;
      w_two += index_[v].spt_d[i];
    }
  }
  if (!quiet) cout << "Avg Label size: " << s_spt / V << " + " << s_inv / V << " + " << s_two / V << endl;
  if (!quiet) cout << "Avg 2-hop size: " << w_two / n_two << " (" << n1/n_two << ", " << n2/n_two << ", " << n3/n_two << ", " << n4/n_two << ", " << n5/n_two << ", " << n6/n_two << ", " << n7/n_two << ")" << endl;
  return (s_spt + s_inv + s_two) / V;
}

#endif  // PRUNED_LANDMARK_LABELING_H_
