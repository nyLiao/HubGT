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
  int ConstructTwoHop(const uint32_t r, vector<uint8_t> &tmp_d,
    const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_out,
    const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_inv,
    const vector<pair<vector<uint64_t>, vector<uint64_t>>> &tmp_tri);
  int ConstructTwoHopLoop(uint32_t st, uint32_t ed,
    const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_out,
    const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_inv,
    const vector<pair<vector<uint64_t>, vector<uint64_t>>> &tmp_tri);
  int ConstructTwoHopParallel(
    const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_out,
    const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_inv,
    const vector<pair<vector<uint64_t>, vector<uint64_t>>> &tmp_tri);
  // node input named by `s` is external id, `v` is internal id
  int FetchNode(const int s, vector<int> &node, vector<int> &dist);
  int FetchLoop(const vector<int> &ns, size_t st, size_t ed, vector<int> &node, vector<int> &dist);
  int FetchParallel(const vector<int> &ns, vector<int> &node, vector<int> &dist);

  inline int QueryDistanceTri(const int r, const vector<int> &nw, const vector<int> &ids, vector<int> &ret);
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
    size_t   tail_out, tail_in;
    uint32_t *spt_v;                // PLL Shortest Path nodes (only smaller ids, sorted) | Inverse nodes (only larger ids, not sorted)
    uint8_t  *spt_d;                // PLL Shortest Path distances
    pair<uint64_t, uint64_t> *spt_s;
  } __attribute__((aligned(64)));   // Aligned for cache lines

  size_t V, E;
  bool quiet = false;
  index_t *index_;
  vector<unordered_map<uint32_t, uint8_t>> indexdct_; // only smaller ids
  vector<vector<uint32_t> > adj;
  vector<uint32_t> alias, alias_inv;

  inline void Init();
  void Free();

  inline int Global(const int v, vector<int> &node, vector<int> &dist, vector<int> &ids);
  inline int Label(const int v, vector<int> &node, vector<int> &dist, vector<int> &ids);
  inline int InvLabel(const int v, vector<int> &node, vector<int> &dist, vector<int> &ids);
  inline int SNeighbor(const int v, const int size, vector<int> &node, vector<int> &dist, vector<int> &ids);
  inline int SampleSet(const int size,
    const vector<int> &set, const vector<int> &set2, const vector<int> &set3,
    vector<int> &ret, vector<int> &ret2, vector<int> &ret3);

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
  vector<pair<uint64_t, uint64_t> > tmp_s(V+1);
  vector<uint8_t> tmp_d(V+1);
  vector<pair<uint32_t, uint32_t> > sibling_es(E);
  vector<pair<uint32_t, uint32_t> > child_es(E);
  {
    vector<uint32_t> que(V);

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
        tmp_out(V, make_pair(vector<uint32_t>(1, V), vector<uint8_t>(1, INF8)));
    vector< pair<vector<uint32_t>, vector<uint8_t>> >
        tmp_inv(V, make_pair(vector<uint32_t>(), vector<uint8_t>()));
    vector< pair<vector<uint64_t>, vector<uint64_t>> >
        tmp_tri(V, make_pair(vector<uint64_t>(1, 0), vector<uint64_t>(1, 0)));

    vector<bool> vis(V);
    vector<uint32_t> que(V);
    vector<uint8_t> tmp_dr(V+1, INF8);

    time_search = -GetCurrentTimeSec();
    for (size_t r = 0; r < V; ++r) {
      if (usd[r]) continue;
      const index_t &idx_r = index_[r];
      const pair<vector<uint32_t>, vector<uint8_t>> &tmp_out_r = tmp_out[r];
      pair<vector<uint32_t>, vector<uint8_t>> &tmp_inv_r = tmp_inv[r];
      fill(tmp_d.begin(), tmp_d.end(), INF8);
      fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

      for (size_t i = 0; i < tmp_out_r.first.size(); ++i) {
        tmp_d[tmp_out_r.first[i]] = tmp_out_r.second[i];
        tmp_dr[tmp_out_r.first[i]] = tmp_out_r.second[i];
      }
      int nns = 0;
      for (size_t i = 0; i < adj[r].size(); ++i) {
        const uint32_t v = adj[r][i];
        if (!vis[v]) {
          tmp_d[v] = 1;
          tmp_s[v].first = 1ULL << nns;
          if (++nns == MAXIDX) break;
        }
      }

      int que_t0 = 0, que_t1 = 1, que_h = 1;
      que[0] = r;
      vis[r] = true;

      for (uint8_t d = 0; que_t0 < que_h && d < MAXDIST; ++d) {
        size_t num_sibling_es = 0, num_child_es = 0;

        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          const uint32_t v = que[que_i];    // v >= r
          pair<vector<uint32_t>, vector<uint8_t>> &tmp_out_v = tmp_out[v];
          const index_t &idx_v = index_[v];

          // Prefetch
          _mm_prefetch(&idx_v.bpspt_d[0], _MM_HINT_T0);
          _mm_prefetch(&idx_v.bpspt_s[0][0], _MM_HINT_T0);
          _mm_prefetch(&tmp_out_v.first[0], _MM_HINT_T0);
          _mm_prefetch(&tmp_out_v.second[0], _MM_HINT_T0);

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
          for (size_t i = 0; i < tmp_out_v.first.size(); ++i) {
            const uint32_t w = tmp_out_v.first[i];
            const uint8_t td = tmp_out_v.second[i] + tmp_dr[w];
            if (td <= d) goto pruned;
          }

          // Traverse
          tmp_d[v] = d;
          tmp_out_v.first .back() = r;
          tmp_out_v.second.back() = d;
          tmp_out_v.first .emplace_back(V);
          tmp_out_v.second.emplace_back(INF8);
          if (d > 0 && d < HMAXDIST) {
            tmp_inv_r.first .emplace_back(v);
            tmp_inv_r.second.emplace_back(d);
          }
          for (size_t i = 0; i < adj[v].size(); ++i) {
            const uint32_t w = adj[v][i];
            if (!vis[w] && tmp_out[w].first.size() < MAXIDX) {
              vis[w] = true;
              if (usd[w]) continue;

              if (d == tmp_d[w]) {
                if (v < w) {
                  sibling_es[num_sibling_es].first  = v;
                  sibling_es[num_sibling_es].second = w;
                  ++num_sibling_es;
                }
              } else if (d < tmp_d[w]) {
                child_es[num_child_es].first  = v;
                child_es[num_child_es].second = w;
                ++num_child_es;
              }

              que[que_h++] = w;
            }
          }
        pruned:
          {}
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
      } // tmp_out[r] & tmp_inv[r] are finished

      pair<vector<uint64_t>, vector<uint64_t>> &tmp_tri_r = tmp_tri[r];
      for (uint32_t v: tmp_out_r.first) {
        if (v == V) continue;
        tmp_tri_r.first .back() = tmp_s[v].first;
        tmp_tri_r.second.back() = tmp_s[v].second & ~tmp_s[v].first;
        tmp_tri_r.first .emplace_back(0);
        tmp_tri_r.second.emplace_back(0);
      }
      for (uint32_t v: tmp_inv_r.first) {
        tmp_tri_r.first .emplace_back(tmp_s[v].first);
        tmp_tri_r.second.emplace_back(tmp_s[v].second & ~tmp_s[v].first);
      }

      for (int i = 0; i < que_h; ++i) vis[que[i]] = false;
      for (size_t i = 0; i < tmp_out_r.first.size(); ++i) {
        tmp_dr[tmp_out_r.first[i]] = INF8;
      }
      usd[r] = true;

      if (!quiet && r % (V / 20) == 0){
        cout << time_search+GetCurrentTimeSec() << " (" << (100 * r / V) << "%) " << std::flush;
      }
    }
    time_search += GetCurrentTimeSec();
    if (!quiet) cout << "| Search time: " << time_search << endl;

    time_two = -GetCurrentTimeSec();
    ConstructTwoHopParallel(tmp_out, tmp_inv, tmp_tri);
    time_two += GetCurrentTimeSec();

    for (size_t v = 0; v < V; ++v) {
      const size_t sz_out = tmp_out[v].first.size();
      const size_t sz_inv = tmp_inv[v].first.size();
      const size_t sz_all = sz_out + sz_inv;
      index_[v].tail_out = sz_out;
      index_[v].tail_in = sz_all;

      index_[v].spt_v = (uint32_t*)memalign(64, sz_all * sizeof(uint32_t));
      index_[v].spt_d = (uint8_t *)memalign(64, sz_all * sizeof(uint8_t ));
      index_[v].spt_s = (pair<uint64_t, uint64_t>*)memalign(64, sz_all * sizeof(pair<uint64_t, uint64_t>));
      for (size_t i = 0; i < sz_out; ++i) index_[v].spt_v[i] = tmp_out[v].first[i];
      for (size_t i = 0; i < sz_out; ++i) index_[v].spt_d[i] = tmp_out[v].second[i];
      for (size_t i = 0; i < sz_inv; ++i) index_[v].spt_v[sz_out+i] = tmp_inv[v].first[i];
      for (size_t i = 0; i < sz_inv; ++i) index_[v].spt_d[sz_out+i] = tmp_inv[v].second[i];
      for (size_t i = 0; i < sz_all; ++i) {
        index_[v].spt_s[i].first  = tmp_tri[v].first[i];
        index_[v].spt_s[i].second = tmp_tri[v].second[i];
      }

      tmp_out[v].first.clear();
      tmp_out[v].second.clear();
      tmp_inv[v].first.clear();
      tmp_inv[v].second.clear();
      tmp_tri[v].first.clear();
      tmp_tri[v].second.clear();
    }
  }

  if (!quiet) cout << "| Two-hop time: " << time_two << endl;
  if (!quiet) AverageLabelSize();
  return time_neighbor + time_search + time_two;
}

int PrunedLandmarkLabeling::
ConstructTwoHop(const uint32_t r, vector<uint8_t> &tmp_d,
  const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_out,
  const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_inv,
  const vector<pair<vector<uint64_t>, vector<uint64_t>>> &tmp_tri) {

  const pair<vector<uint32_t>, vector<uint8_t>> &tmp_out_r = tmp_out[r];
  const pair<vector<uint32_t>, vector<uint8_t>> &tmp_inv_r = tmp_inv[r];
  unordered_map<uint32_t, uint8_t> &dct_r = indexdct_[r];
  queue<uint32_t> que_two;

  _mm_prefetch(&tmp_out_r.first[0], _MM_HINT_T0);
  _mm_prefetch(&tmp_out_r.second[0], _MM_HINT_T0);
  _mm_prefetch(&tmp_inv_r.first[0], _MM_HINT_T0);
  _mm_prefetch(&tmp_inv_r.second[0], _MM_HINT_T0);

  dct_r.reserve((tmp_out_r.first.size() + 1) * (tmp_inv_r.first.size() + 1));
  for (size_t i = 0; i < tmp_out_r.first.size(); ++i) {
    dct_r.emplace(tmp_out_r.first[i], tmp_out_r.second[i]);
  }
  for (size_t i = 0; i < tmp_inv_r.first.size(); ++i) {
    dct_r.emplace(tmp_inv_r.first[i], tmp_inv_r.second[i]);
  }

  // 2-hop neighbors
  for (size_t i = 0; ; ++i) {
    const uint32_t v = tmp_out_r.first[i];
    if (v >= r) break;
    const uint8_t d_rv = tmp_out_r.second[i];
    if (d_rv > HMAXDIST-2) continue;
    const pair<vector<uint32_t>, vector<uint8_t>> &tmp_inv_v = tmp_inv[v];
    const pair<vector<uint64_t>, vector<uint64_t>> &tmp_tri_v = tmp_tri[v];
    const auto foundIt = std::find(tmp_inv_v.first.begin(), tmp_inv_v.first.end(), r);
    const int ir = (foundIt == tmp_inv_v.first.end()) ? -1 : foundIt - tmp_inv_v.first.begin() + tmp_out[v].first.size();
    const uint64_t s_rv0 = tmp_tri_v.first[ir], s_rv1 = tmp_tri_v.second[ir];

    _mm_prefetch(&tmp_inv_v.first[0], _MM_HINT_T0);
    _mm_prefetch(&tmp_inv_v.second[0], _MM_HINT_T0);

    for (size_t j = 0; j < tmp_inv_v.first.size(); ++j) {
      const uint32_t w = tmp_inv_v.first[j];
      if (w >= r) continue;

      uint8_t td = d_rv + tmp_inv_v.second[j];
      if (td > HMAXDIST) continue;

      if (td - 2 <= tmp_d[w]) {
        const size_t iv = j + tmp_out[v].first.size();
        td +=
            (s_rv0 & tmp_tri_v.first[iv]) ? -2 :
            ((s_rv0 & tmp_tri_v.second[iv]) | (s_rv1 & tmp_tri_v.first[iv]))
            ? -1 : 0;

        if (td < tmp_d[w]) {
          if (tmp_d[w] < MAXDIST) {
            que_two.push(w);
          }
          tmp_d[w] = td;
        }
      }
    }
  }

  while (!que_two.empty()) {
    uint32_t w = que_two.front();
    que_two.pop();
    if (tmp_d[w] >= MAXDIST) continue;

    auto entry_w = dct_r.find(w);
    if (entry_w == dct_r.end()) {
      dct_r.emplace(w, tmp_d[w]);
    } else if (tmp_d[w] < entry_w->second) {
      entry_w->second = tmp_d[w];
    }
    tmp_d[w] = INF8;
  }
  return dct_r.size();
}

int PrunedLandmarkLabeling::
ConstructTwoHopLoop(uint32_t st, uint32_t ed,
  const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_out,
  const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_inv,
  const vector<pair<vector<uint64_t>, vector<uint64_t>>> &tmp_tri) {

  vector<uint8_t> tmp_d(V+1, INF8);
  for (uint32_t r = st; r < ed; ++r) {
    ConstructTwoHop(r, tmp_d, tmp_out, tmp_inv, tmp_tri);
    if (!quiet && st == 0 && r % (ed / 20) == 0){
      cout << r << " (" << (100 * r / ed) << "%) " << std::flush;
    }
  }
  return ed - st;
}

int PrunedLandmarkLabeling::
ConstructTwoHopParallel(
  const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_out,
  const vector<pair<vector<uint32_t>, vector<uint8_t>>> &tmp_inv,
  const vector<pair<vector<uint64_t>, vector<uint64_t>>> &tmp_tri) {

  vector<std::thread> threads;
  uint32_t st, ed = 0;
  uint32_t it;

  for (it = 1; it <= V % NUMTHREAD; it++) {
    st = ed;
    ed += std::ceil((float)V / NUMTHREAD);
    threads.push_back(std::thread(&PrunedLandmarkLabeling::ConstructTwoHopLoop, this, st, ed, ref(tmp_out), ref(tmp_inv), ref(tmp_tri)));
  }
  for (; it <= NUMTHREAD; it++) {
    st = ed;
    ed += V / NUMTHREAD;
    threads.push_back(std::thread(&PrunedLandmarkLabeling::ConstructTwoHopLoop, this, st, ed, ref(tmp_out), ref(tmp_inv), ref(tmp_tri)));
  }
  for (size_t t = 0; t < NUMTHREAD; t++)
    threads[t].join();
  vector<std::thread>().swap(threads);
  return V;
}

// ====================
int PrunedLandmarkLabeling::
FetchNode(const int s, vector<int> &node, vector<int> &dist){
  const uint32_t v = alias[s];
  const index_t &idx_v = index_[v];
  size_t s_bp, s_spt, s_inv, s_adj, sn_adj, s_total;

  vector<int> tmp_node(4*MAXIDX), tmp_dist(4*MAXIDX);
  vector<int> ids(4*MAXIDX), tmp_ids(4*MAXIDX);
  vector<vector<int>> mat(NUM_FETCH, vector<int>{});
  vector<int> &mat0 = mat[0];

  // Construct sample
  node.clear();
  node.reserve(NUM_FETCH);
  ids.clear();
  ids.reserve(NUM_FETCH);
  node.emplace_back(v);
  mat0.emplace_back(0);
  ids.emplace_back(-1);

  Global(v, tmp_node, tmp_dist, tmp_ids);
  s_bp = SampleSet(NUM_BP, tmp_node, tmp_dist, tmp_ids, node, mat0, ids);
  Label(v, tmp_node, tmp_dist, tmp_ids);
  s_spt = SampleSet(NUM_SPT, tmp_node, tmp_dist, tmp_ids, node, mat0, ids);
  InvLabel(v, tmp_node, tmp_dist, tmp_ids);
  s_inv = SampleSet(NUM_INV, tmp_node, tmp_dist, tmp_ids, node, mat0, ids);
  s_total = s_bp + s_spt + s_inv;
  sn_adj = NUM_FETCH - s_total - 1;
  SNeighbor(v, sn_adj, tmp_node, tmp_dist, tmp_ids);
  s_adj = SampleSet(sn_adj, tmp_node, tmp_dist, tmp_ids, node, mat0, ids);
  s_total += s_adj + 1;

  auto endIt = idx_v.spt_v + idx_v.tail_in;
  for (size_t i = 0; i < s_total; ++i)
    if (ids[i] == -1) {
      auto findIt = std::find(idx_v.spt_v, endIt, node[i]);
      if (findIt != endIt) ids[i] = findIt - idx_v.spt_v;
    }

  // Query pair-wise distance: node (len s) -> dist (s * s)
  vector<int> argnode(s_total);
  std::iota(argnode.begin(), argnode.end(), 0);
  std::sort(argnode.begin(), argnode.end(), [&node](int a, int b) {
    return node[a] > node[b];
  });

  tmp_node.clear();
  tmp_ids.clear();
  tmp_node.emplace_back(node[argnode[s_total-1]]);
  tmp_ids.emplace_back(ids[argnode[s_total-1]]);
  for (size_t j = 1; j < s_total; ++j) {
    mat[j].resize(s_total);
    mat[j][0] = mat0[j];
  }

  for (int i = s_total-2; i >= 0; --i) {
    tmp_node.emplace_back(node[argnode[i]]);
    tmp_ids.emplace_back(ids[argnode[i]]);
    if (node[argnode[i]] == v) continue;
    tmp_dist.clear();
    QueryDistanceTri(v, tmp_node, tmp_ids, tmp_dist);
    // mat[argnode[i]][argnode[i]] = 0;
    for (int j = s_total-1; j > i; --j) {
      mat[argnode[i]][argnode[j]] = tmp_dist[s_total-1 - j];
      mat[argnode[j]][argnode[i]] = tmp_dist[s_total-1 - j];
    }
  }
  for (size_t i = 0; i < s_total; ++i) {
      // TODO: return global nodes
      node[i] = alias_inv[node[i]];
  }

  // Align to output
  // node and mat[i]: s -> n
  while (node.size() < NUM_FETCH) {
    const size_t copy_size = std::min(s_total, NUM_FETCH - node.size());
    node.insert(node.end(), node.begin(), node.begin() + copy_size);
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
FetchLoop(const vector<int> &ns, size_t st, size_t ed, vector<int> &node, vector<int> &dist){
  vector<int> tmp_node, tmp_dist;
  for (size_t i = st; i < ed; i++){
    FetchNode(ns[i], tmp_node, tmp_dist);
    std::copy(tmp_node.begin(), tmp_node.end(), node.begin() + i * NUM_FETCH);
    std::copy(tmp_dist.begin(), tmp_dist.end(), dist.begin() + i * NUM_FETCH * NUM_FETCH);
  }
  return (ed - st);
}

int PrunedLandmarkLabeling::
FetchParallel(const vector<int> &ns, vector<int> &node, vector<int> &dist){
  vector<std::thread> threads;
  size_t st, ed = 0;
  size_t it;

  for (it = 1; it <= ns.size() % NUMTHREAD; it++) {
    st = ed;
    ed += std::ceil((float)ns.size() / NUMTHREAD);
    threads.push_back(std::thread(&PrunedLandmarkLabeling::FetchLoop, this, ref(ns), st, ed, ref(node), ref(dist)));
  }
  for (; it <= NUMTHREAD; it++) {
    st = ed;
    ed += ns.size() / NUMTHREAD;
    threads.push_back(std::thread(&PrunedLandmarkLabeling::FetchLoop, this, ref(ns), st, ed, ref(node), ref(dist)));
  }
  for (size_t t = 0; t < NUMTHREAD; t++)
    threads[t].join();
  vector<std::thread>().swap(threads);
  return node.size();
}

// ====================
int PrunedLandmarkLabeling::
Global(const int v, vector<int> &node, vector<int> &dist, vector<int> &ids){
  node.clear();
  dist.clear();
  ids.clear();
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
      node.emplace_back(alias[V+i]);
      dist.emplace_back(d);
      ids.emplace_back(-1);
    }
  }
  return node.size();
}

int PrunedLandmarkLabeling::
Label(const int v, vector<int> &node, vector<int> &dist, vector<int> &ids){
  node.clear();
  dist.clear();
  ids.clear();
  const index_t &idx_v = index_[v];
  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);

  for (size_t i = 0; idx_v.spt_v[i] != V; ++i){
    if (idx_v.spt_d[i] == 0) continue;
    node.emplace_back(idx_v.spt_v[i]);
    dist.emplace_back(idx_v.spt_d[i]);
    ids.emplace_back(i);
  }
  return node.size();
}

int PrunedLandmarkLabeling::
InvLabel(const int v, vector<int> &node, vector<int> &dist, vector<int> &ids){
  node.clear();
  dist.clear();
  ids.clear();
  const index_t &idx_v = index_[v];
  _mm_prefetch(&idx_v.spt_v[idx_v.tail_out], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[idx_v.tail_out], _MM_HINT_T0);

  for (size_t i = idx_v.tail_out; i < idx_v.tail_in; ++i){
    if (idx_v.spt_d[i] == 0) continue;
    node.emplace_back(idx_v.spt_v[i]);
    dist.emplace_back(idx_v.spt_d[i]);
    ids.emplace_back(i);
  }
  return node.size();
}

int PrunedLandmarkLabeling::
SNeighbor(const int v, const int size, vector<int> &node, vector<int> &dist, vector<int> &ids){
  node.clear();
  dist.clear();
  ids.clear();

  queue<uint32_t> node_que, dist_que;
  vector<bool> vis(V, false);
  int d_last = 0;
  node_que.push(v);
  dist_que.push(0);
  vis[v] = true;

  while (!node_que.empty() && node.size() < size_t(4*size*size)){
    uint32_t u = node_que.front();
    uint8_t  d = dist_que.front();
    node_que.pop();
    dist_que.pop();
    // Exit condition for every hop
    if (d > d_last && node.size() >= size_t(size)) break;
    d_last = d;

    if (adj[u].size() > MAXIDX){
      size_t i;
      for (size_t ii = 0; i < MAXIDX; ii++){
        i = rand() % adj[u].size();
        if (vis[adj[u][i]]) continue;
        node_que.push(adj[u][i]);
        dist_que.push(d + 1);
        node.emplace_back(adj[u][i]);
        dist.emplace_back(d + 1);
        ids.emplace_back(-1);
        vis[adj[u][i]] = true;
      }
    }
    else{
      for (size_t i = 0; i < adj[u].size(); i++){
        if (vis[adj[u][i]]) continue;
        node_que.push(adj[u][i]);
        dist_que.push(d + 1);
        node.emplace_back(adj[u][i]);
        dist.emplace_back(d + 1);
        ids.emplace_back(-1);
        vis[adj[u][i]] = true;
      }
    }
  }

  return node.size();
}

int PrunedLandmarkLabeling::
SampleSet(const int size,
  const vector<int> &set, const vector<int> &set2, const vector<int> &set3,
  vector<int> &ret, vector<int> &ret2, vector<int> &ret3) {

  if (set.size() <= size) {
    ret.insert (ret.end(),  std::make_move_iterator(set.begin()),
                            std::make_move_iterator(set.end()));
    ret2.insert(ret2.end(), std::make_move_iterator(set2.begin()),
                            std::make_move_iterator(set2.end()));
    ret3.insert(ret3.end(), std::make_move_iterator(set3.begin()),
                            std::make_move_iterator(set3.end()));
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
      ret3.emplace_back(set3[index]);
      cnt++;
    }
  }
  return size;
}

// ====================
// Query between nw[-1] and nw[0..sz-2] orienting r
int PrunedLandmarkLabeling::
QueryDistanceTri(const int r, const vector<int> &nw, const vector<int> &ids, vector<int> &ret) {
  const size_t sz = nw.size();
  const int v = nw[sz-1];
  const int iv = ids[sz-1];
  const index_t &idx_r = index_[r];
  const unordered_map<uint32_t, uint8_t> &dct_v = indexdct_[v];
  const uint64_t s_rv0 = (iv != -1) ? idx_r.spt_s[iv].first : 0;
  const uint64_t s_rv1 = (iv != -1) ? idx_r.spt_s[iv].second : 0;

  _mm_prefetch(&idx_r.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_r.spt_d[0], _MM_HINT_T0);

  for (size_t j = 0; j < sz-1; ++j) {
    const int w = nw[j];
    if (w >= v) {
      ret[j] = 0;
      continue;
    }

    auto it = dct_v.find(nw[j]);
    if (it != dct_v.end()) {
      ret[j] = it->second;
      continue;
    }

    uint8_t d = INF8;
    const int iw = ids[j];
    if (iv != -1 && iw != -1) {
      d = idx_r.spt_d[iv] + idx_r.spt_d[iw]
        + ((s_rv0 & idx_r.spt_s[iw].first) ? -2 :
          ((s_rv0 & idx_r.spt_s[iw].second) | (s_rv1 & idx_r.spt_s[iw].first))
          ? -1 : 0);
    } else {
      d = QueryDistance(v, w);
    }

    ret[j] = d;
  }
  return sz;
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

  if (d == INF8) {
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
    unordered_map<uint32_t, uint8_t> &dct_v = indexdct_[v];
    dct_v.emplace(w, d);
  }
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

    WRITE_BINARY(idx.tail_out);
    WRITE_BINARY(idx.tail_in);
    for (uint32_t i = 0; i < idx.tail_in; ++i) {
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

    READ_BINARY(idx.tail_out);
    READ_BINARY(idx.tail_in);
    idx.spt_v = (uint32_t*)memalign(64, idx.tail_in * sizeof(uint32_t));
    idx.spt_d = (uint8_t *)memalign(64, idx.tail_in * sizeof(uint8_t ));
    for (uint32_t i = 0; i < idx.tail_in; ++i) {
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
  indexdct_.resize(V);
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
  indexdct_.clear();
  V = 0;
  E = 0;
}

double PrunedLandmarkLabeling::
AverageLabelSize() {
  double s_spt = 0.0, s_inv = 0.0, s_two = 0.0;
  double w_two = 0.0, n_two = 0.0;
  double n1 = 0.0, n2 = 0.0, n3 = 0.0, n4 = 0.0, n5 = 0.0, n6 = 0.0, n7 = 0.0;
  for (size_t v = 0; v < V; ++v) {
    s_spt += index_[v].tail_out;
    s_inv += index_[v].tail_in - index_[v].tail_out;
    s_two += indexdct_[v].size();
    for (auto it = indexdct_[v].begin(); it != indexdct_[v].end(); ++it) {
      switch (it->second) {
        case 1: n1++; break;
        case 2: n2++; break;
        case 3: n3++; break;
        case 4: n4++; break;
        case 5: n5++; break;
        case 6: n6++; break;
        default: n7++; break;
      }
      n_two++;
      w_two += it->second;
    }
  }
  if (!quiet) cout << "Avg Label size: " << s_spt / V << " + " << s_inv / V << " + " << s_two / V << endl;
  if (!quiet) cout << "Avg 2-hop size: " << w_two / n_two << " (" << n1/n_two << ", " << n2/n_two << ", " << n3/n_two << ", " << n4/n_two << ", " << n5/n_two << ", " << n6/n_two << ", " << n7/n_two << ")" << endl;
  return (s_spt + s_inv + s_two) / V;
}

#endif  // PRUNED_LANDMARK_LABELING_H_
