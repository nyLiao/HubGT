#include "kpll.hpp"
#include <iostream>
#include <climits>
// #include <xmmintrin.h>
#include <cassert>
#include <cstdlib>
#include <queue>
#include <algorithm>
#include <memory.h>
using namespace std;

const uint8_t TopKPrunedLandmarkLabeling::INF8 = std::numeric_limits<uint8_t>::max() / 2;

template <typename T> inline bool ReAlloc(T*& ptr, size_t nmemb){
  ptr = (T*)realloc(ptr, nmemb * sizeof(T));
  return ptr != NULL;
}

template <typename T> inline void EraseVector(std::vector<T> &vec){
  std::vector<T>().swap(vec);
}


double GetCurrentTimeSec(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// the address of the distance array of label l
inline uint8_t* TopKPrunedLandmarkLabeling::index_t::
GetDistArray(size_t i) const {
  size_t off = i & dist_array_t::mask;
  const dist_array_t &dn = d_array[i >> dist_array_t::width];
  return dn.addr + (off == 0 ? 0 : dn.offset[off - 1]);
}

//the length of the distance array of label l
inline uint8_t TopKPrunedLandmarkLabeling::index_t::
DistArrayLength(size_t i) const {
  size_t off = i & dist_array_t::mask;
  const dist_array_t &da = d_array[i >> dist_array_t::width];
  return off == 0 ? da.offset[0] : da.offset[off] - da.offset[off - 1];
}

inline bool TopKPrunedLandmarkLabeling::index_t::
ReAllocDistArray(size_t i, size_t nmemb){
  size_t off = i & dist_array_t::mask;
  dist_array_t &da = d_array[i >> dist_array_t::width];
  size_t n = nmemb + (off == 0 ? 0 : da.offset[off-1]);

  if(da.offset[off] < n){
    da.offset[off] = n;
    return ReAlloc(da.addr, n);
  }else{
    assert(da.addr[da.offset[off] - 1] == std::numeric_limits<uint8_t>::max() / 2);
    return true;
  }
}

float TopKPrunedLandmarkLabeling::
ConstructIndex(const vector<uint32_t> &ns, const vector<uint32_t> &nt, size_t K, bool directed, bool quiet){
  Free();

  this->V = 0;
  this->K = K;
  this->directed = directed;
  this->quiet = quiet;
  V = std::max(*max_element(ns.begin(), ns.end()), *max_element(nt.begin(), nt.end())) + 1;
  if (!quiet) cout << "Nodes: " << V << ", Edges: " << ns.size() << ", K: " << K << ", Directed: " << directed << endl;

  for (int dir = 0; dir < 1 + directed; dir++){
    graph[dir].resize(V);
  }

  // renaming
  vector<std::pair<uint32_t, uint32_t> > deg(V, std::make_pair(0, 0));

  for (size_t i = 0; i < V; i++) deg[i].second = i;
  for (size_t i = 0; i < ns.size(); i++){
    deg[ns[i]].first++;
    deg[nt[i]].first++;
  }

  sort(deg.begin(), deg.end(), greater<pair<uint32_t, uint32_t> >());
  alias.resize(V);
  alias_inv.resize(V);

  for (size_t i = 0; i < V; i++) {
    alias[deg[i].second] = i;
    alias_inv[i] = deg[i].second;
  }

  for (size_t i = 0; i < ns.size(); i++){
    graph[0][alias[ns[i]]].push_back(alias[nt[i]]);

    if (directed){
      graph[1][alias[nt[i]]].push_back(alias[ns[i]]);
    } else {
      graph[0][alias[nt[i]]].push_back(alias[ns[i]]);
    }
  }

  Init();

  bool status = true;
  loop_count_time = -GetCurrentTimeSec();
  for(size_t v = 0; v < V; v++){
    CountLoops(v, status);
  }
  loop_count_time += GetCurrentTimeSec();

  indexing_time = -GetCurrentTimeSec();
  for(size_t v = 0; v < V; v++){
    // if (deg[v].first <= 2) break;
    PrunedBfs(v, false, status);
    if (directed){
      PrunedBfs(v, true, status);
    }

    if (!quiet && v % (V / 10) == 0){
      cout << indexing_time+GetCurrentTimeSec() << " (" << (100 * v / V) << "%) ";
    }
  }
  indexing_time += GetCurrentTimeSec();

  if (!quiet) cout << endl << "Loop count time: " << loop_count_time << ", Indexing time: " << indexing_time << endl;
  return loop_count_time + indexing_time;
}

TopKPrunedLandmarkLabeling::
~TopKPrunedLandmarkLabeling(){
  Free();
}


int TopKPrunedLandmarkLabeling::
KDistanceQuery(int s, int t, uint8_t k){
  vector<int> dists;
  return KDistanceQuery(s, t, k, dists);
}

int TopKPrunedLandmarkLabeling::
KDistanceQuery(int s, int t, uint8_t k, vector<int> &ret){
  ret.clear();
  s = alias[s];
  t = alias[t];
  size_t pos1 = 0;
  size_t pos2 = 0;

  vector<int> count(30, 0);
  // cerr << directed << " " << s << " " << t << endl;
  const index_t &ids = index[directed][s];
  const index_t &idt = index[0][t];

  uint32_t *ls = ids.label;
  uint32_t *lt = idt.label;

  for (;;){
    if (ls[pos1] == lt[pos2]){
      uint32_t W = ls[pos1];
      if (W == V) break;

      uint8_t *dcs = ids.GetDistArray(pos1);
      uint8_t *dct = idt.GetDistArray(pos2);

      for (size_t i = 0; dcs[i] != INF8; i++){
        for (size_t j = 0; dct[j] != INF8; j++){
          for (size_t m = 0; m < loop_count[W].size(); m++){
            uint8_t d_tmp = ids.offset[pos1] + idt.offset[pos2] + i + j + m;
            uint8_t c_tmp = loop_count[W][m] - (m ? loop_count[W][m-1] : 0);
            if (count.size() <= d_tmp) count.resize(d_tmp + 1, 0);
            count[d_tmp] += (int)dcs[i] * dct[j] * c_tmp;
          }
        }
      }
      pos1++, pos2++;
    } else {
      if (ls[pos1] < lt[pos2]){
        pos1++;
      } else {
        pos2++;
      }
    }
  }

  for (size_t i = 0; i < count.size(); i++){
    while (ret.size() < k && count[i]-- > 0){
      ret.push_back(i);
    }
  }

  return ret.size() < k ? INT_MAX : 0;
}

int TopKPrunedLandmarkLabeling::
Label(int v, vector<int> &pos, vector<int> &dist){
  pos.clear();
  dist.clear();

  const index_t &idv = index[directed][alias[v]];

  uint32_t *lb = idv.label;
  for (size_t i = 0; lb[i] != V; i++){
    if (lb[i] == alias[v]) continue;
    pos.push_back(alias_inv[lb[i]]);
    dist.push_back(idv.offset[i]);
  }

  return pos.size();
}

int TopKPrunedLandmarkLabeling::
SNeighbor(int v, int size, vector<int> &pos, vector<int> &dist){
  pos.clear();
  dist.clear();

  const vector<vector<uint32_t> > &fgraph = graph[directed];
  std::queue<uint32_t> node_que, dist_que;
  std::vector<bool> updated(V, false);
  int d_last = 0;
  node_que.push(alias[v]);
  dist_que.push(0);
  updated[alias[v]] = true;

  while (!node_que.empty()){
    uint32_t u = node_que.front();
    uint8_t  d = dist_que.front();
    node_que.pop();
    dist_que.pop();
    // Exit condition for every hop
    if (d > d_last && pos.size() >= size) break;
    d_last = d;

    for (size_t i = 0; i < fgraph[u].size(); i++){
      if (updated[fgraph[u][i]]) continue;
      node_que.push(fgraph[u][i]);
      dist_que.push(d + 1);
      pos.push_back(alias_inv[fgraph[u][i]]);
      dist.push_back(d + 1);
      updated[fgraph[u][i]] = true;
    }
  }

  return pos.size();
}

int TopKPrunedLandmarkLabeling::
SPush(int v, int size, float alpha, vector<int> &pos, vector<float> &dist){
  pos.clear();
  dist.clear();

  const vector<vector<uint32_t> > &fgraph = graph[directed];
  std::queue<uint32_t> node_que;
  std::queue<float>    dist_que;
  std::vector<bool> updated(V, false);
  int count = 0;
  node_que.push(alias[v]);
  dist_que.push(1.0);
  updated[alias[v]] = true;

  while (!node_que.empty()){
    uint32_t u = node_que.front();
    float    d = dist_que.front();
    node_que.pop();
    dist_que.pop();
    if (count >= size) break;

    for (size_t i = 0; i < fgraph[u].size(); i++){
      if (!updated[fgraph[u][i]]) count++;
      node_que.push(fgraph[u][i]);
      dist_que.push(d * alpha);
      pos.push_back(alias_inv[fgraph[u][i]]);
      dist.push_back(d * (1 - alpha) / fgraph[u].size());
      updated[fgraph[u][i]] = true;
    }
  }

  return pos.size();
}

size_t TopKPrunedLandmarkLabeling::
IndexSize(){
  size_t sz = 0;

  sz += sizeof(int) * alias.size();		      // alias

  for(size_t v = 0; v < V; v++){
    sz += sizeof(uint8_t ) * loop_count[v].size(); // loopcount

    sz += sizeof(uint32_t) * graph[0][v].size();	  // graph
    sz += sizeof(uint32_t) * graph[1][v].size();
  }

  // index's size
  for (int dir = 0; dir < 1 + directed; dir++){
    for(size_t v = 0; v < V; v++){
      sz += sizeof(index_t);
      sz += index[dir][v].length * sizeof(uint32_t); // index[i].label
      sz += index[dir][v].length * sizeof(uint8_t ); // index[i].offset

      // index[i].d_array
      for(int pos = 0; index[dir][v].label[pos] != V; pos++){
        if((pos & dist_array_t::mask) == 0){
          sz += sizeof(dist_array_t);
        }
        int j = 0;
        do{
          sz += sizeof(uint8_t);
        }while(index[dir][v].GetDistArray(pos)[j++] != INF8);
      }
    }
  }
  return sz;
}

double TopKPrunedLandmarkLabeling::
AverageLabelSize(){
  double total = 0;
  for (int dir = 0; dir < 1 + directed; dir++){
    for (size_t v = 0; v < V; v++){
      total += index[dir][v].length;
    }
  }
  return total / V;
}


void TopKPrunedLandmarkLabeling::
Init(){
  tmp_pruned.resize(V, false);
  tmp_offset.resize(V, INF8);
  tmp_count .resize(V, 0);
  tmp_s_offset.resize(V, INF8); tmp_s_offset.push_back(0);
  tmp_s_count .resize(V);
  for (int j = 0; j < 2; j++) tmp_dist_count[j].resize(V, 0);

  loop_count.resize(V);

  for (int dir = 0; dir < 1 + directed; dir++){
    index[dir] = (index_t*) malloc(sizeof(index_t) * V);

    for (size_t v = 0; v < V; v++){
      index[dir][v].label    = (uint32_t*)malloc(sizeof(uint32_t) * 1);
      index[dir][v].label[0] = V;
      index[dir][v].length   = 0;
      index[dir][v].offset   = NULL;
      index[dir][v].d_array  = NULL;
    }
  }
}

void TopKPrunedLandmarkLabeling::
Free(){
  V = K = loop_count_time = indexing_time = 0;
  directed = false;

  alias.clear();
  alias_inv.clear();
  loop_count.clear();

  for (int i = 0; i < 2; i++){
    graph[i].clear();
  }

  tmp_pruned.clear();
  tmp_offset.clear();
  tmp_count .clear();
  tmp_s_offset.clear();
  tmp_s_count .clear();

  for (int i = 0; i < 2; i++){
    tmp_dist_count[i].clear();
  }


  for (int dir = 0; dir < 1 + directed; dir++){
    for (size_t v = 0; v < V; v++){
      index_t &idv = index[dir][v];

      free(idv.label);

      if (idv.offset != NULL) free(idv.offset);

      if (idv.d_array != NULL){
        for (size_t i = 0; i < idv.length; i += dist_array_t::size){
          free(idv.d_array[i / dist_array_t::size].addr);
        }
        free(idv.d_array);
      }
    }
    free(index[dir]);
  }
}

void TopKPrunedLandmarkLabeling::
CountLoops(uint32_t s, bool &status){
  size_t  count = 0;
  int     curr  = 0;
  int     next  = 1;
  uint8_t dist  = 0;

  std::queue<uint32_t> node_que[2];
  vector<uint32_t>     updated;
  const vector<vector<uint32_t> > &fgraph = graph[0];

  node_que[curr].push(s);
  updated.push_back(s);
  tmp_dist_count[curr][s] = 1;

  for (;;){
    if (dist == INF8 && status){
      cerr << "Warning: Self loops become too long." << endl;
      status = false;
    }

    while (!node_que[curr].empty() && count < K){
      uint32_t v = node_que[curr].front(); node_que[curr].pop();
      uint8_t  c = tmp_dist_count[curr][v]; // the number of path from s to v with dist hops.
      tmp_dist_count[curr][v] = 0;
      if (c == 0) continue;

      if (v == s){
        loop_count[s].resize(dist + 1, 0);
        loop_count[s][dist] += c;
        count += c;
      }

      for (size_t i = 0; i < fgraph[v].size(); i++){
        uint32_t to = fgraph[v][i];

        if (tmp_count[to] == 0){
          updated.push_back(to);
        }

        if (to >= s && tmp_count[to] < K){
          tmp_count[to] += c;
          node_que[next].push(to);
          tmp_dist_count[next][to] += c;
        }
      }
    }
    if(node_que[next].empty() || count >= K) break;
    swap(curr, next);
    dist++;
  }

  for(size_t i = 1; i < loop_count[s].size(); i++){
    loop_count[s][i] += loop_count[s][i-1];
  }
  assert(loop_count[s][0] == 1);
  ResetTempVars(s, updated, false);
}

void TopKPrunedLandmarkLabeling::
PrunedBfs(uint32_t s, bool rev, bool &status){
  SetStartTempVars(s, rev);

  int     curr = 0;
  int     next = 1;
  uint8_t dist = 0;

  std::queue<uint32_t> node_que[2];
  vector<uint32_t>     updated;

  node_que[curr].push(s);
  tmp_dist_count[curr][s] = 1;
  updated.push_back(s);
  const vector<vector<uint32_t> > &graph_ = graph[rev];

  for (;;){
    if (dist == INF8 && status){
      cerr << "Warning: Distance from a source node becomes too long." << endl;
      status = false;
    }


    while (!node_que[curr].empty()){

      uint32_t v = node_que[curr].front(); node_que[curr].pop();
      uint8_t  c = tmp_dist_count[curr][v];
      tmp_dist_count[curr][v] = 0;

      if(c == 0 || tmp_pruned[v]) continue;
      tmp_pruned[v] = Pruning(v, dist, rev);
      // cerr << "Pruning done" << endl;

      if(tmp_pruned[v]) continue;

      if(tmp_offset[v] == INF8){
        // Make new label for a node v
        tmp_offset[v] = dist;
        AllocLabel(v, s, dist, c, rev);
      }else{
        ExtendLabel(v, s, dist, c, rev);
      }

      for(size_t i = 0; i < graph_[v].size(); i++){
        uint32_t to  = graph_[v][i];
        if(tmp_count[to] == 0){
          updated.push_back(to);
        }

        if(to > s && tmp_count[to] < K){
          tmp_count[to] += c;
          node_que[next].push(to);
          tmp_dist_count[next][to] += c;
        }
      }
    }

    if (node_que[next].empty()) break;
    if (dist > 4*K) break;
    swap(curr, next);
    dist++;
  }
  // cerr <<"#visited nodes: " << num_of_labeled_vertices[s] << endl;
  ResetTempVars(s, updated, rev);
};

inline void TopKPrunedLandmarkLabeling::
SetStartTempVars(uint32_t s, bool rev){
  const index_t &ids = index[directed && !rev][s];

  for(size_t pos = 0; ids.label[pos] != V; pos++){
    int w = ids.label[pos];
    tmp_s_offset[w] = ids.offset[pos];

    vector<uint8_t> tmp_v;
    for(size_t i = 0; ids.GetDistArray(pos)[i] != INF8; i++){
      tmp_v.push_back(ids.GetDistArray(pos)[i]);
    }
    tmp_s_count[w].resize(tmp_v.size() + loop_count[w].size() - 1, 0);

    for(size_t i = 0; i < tmp_v.size(); i++){
      for(size_t j = 0; j < loop_count[w].size(); j++){
        tmp_s_count[w][i+j] += tmp_v[i] * loop_count[w][j];
      }
    }
  }
}

inline void TopKPrunedLandmarkLabeling::
ResetTempVars(uint32_t s, const vector<uint32_t> &updated, bool rev){
  // cerr << rev << " " << s << " " << V << endl;
  const index_t &ids = index[directed && !rev][s];

  // cerr << ids.length << " " << ids.label[0] << endl;
  for(size_t pos = 0; ids.label[pos] != V; pos++){
    int w = ids.label[pos];
    tmp_s_offset[w] = INF8;
    tmp_s_count[w].clear();
  }

  for(size_t i = 0; i < updated.size(); i++){
    tmp_count [updated[i]] = 0;
    tmp_offset[updated[i]] = INF8;
    tmp_pruned[updated[i]] = false;
    for(int j = 0; j < 2; j++) tmp_dist_count[j][updated[i]] = 0;
  }
}

inline bool TopKPrunedLandmarkLabeling::
Pruning(uint32_t v,  uint8_t d, bool rev){
  const index_t &idv = index[rev][v];

  // _mm_prefetch(idv.label , _MM_HINT_T0);
  // _mm_prefetch(idv.offset, _MM_HINT_T0);

  size_t pcount = 0;

  // cerr << "Pruning start" << endl;
  for (size_t pos = 0;; pos++){
    uint32_t w = idv.label[pos];

    if (tmp_s_offset[w] == INF8) continue;
    if (w == V) break;

    const vector<uint8_t> &dcs = tmp_s_count[w];
    const uint8_t         *dcv = idv.GetDistArray(pos);

    int l = dcs.size() - 1;
    int c = d - tmp_s_offset[w] - idv.offset[pos];

    // By using precompurted table tmp_s_count, compute the number of path with a single loop.
    for (int i = 0; i <= c && dcv[i] != INF8; i++){
      pcount += (int)dcs[std::min(c - i, l)] * dcv[i];
    }

    if (pcount >= K) return true;
  }
  return false;
}

inline void TopKPrunedLandmarkLabeling::
AllocLabel(uint32_t v, uint32_t start, uint8_t dist, uint8_t count, bool dir){
  index_t &idv = index[dir][v];

  size_t size = ++idv.length;
  size_t last = size - 1;

  ReAlloc(idv.label, size + 1);
  idv.label[last] = start;
  idv.label[size] = V;

  ReAlloc(idv.offset, size);
  idv.offset[last] = dist;

  if (last % dist_array_t::size == 0){
    int ds = (size + dist_array_t::size - 1) / dist_array_t::size;
    int dl = ds - 1;
    ReAlloc(idv.d_array, ds);
    idv.d_array[dl].addr = NULL;
    memset(idv.d_array[dl].offset, 0, sizeof(idv.d_array[dl].offset));
  }

  idv.ReAllocDistArray (last, 2);
  // cerr << (unsigned long long )idv.GetDistArray(last) << " " << (int)idv.DistArrayLength(last) << endl;
  idv.GetDistArray(last)[0] = count;
  idv.GetDistArray(last)[1] = INF8;
}

inline void TopKPrunedLandmarkLabeling::
ExtendLabel(uint32_t v, uint32_t start, uint8_t dist, uint8_t count, bool dir){
  index_t &idv = index[dir][v];

  assert(idv.length > 0);
  size_t last     = idv.length - 1;

  assert(idv.DistArrayLength(last) > 0);
  size_t cur_size = idv.DistArrayLength(last);

  assert(dist >= tmp_offset[v]);
  size_t new_size = dist - tmp_offset[v] + 2;

  assert(idv.label[last] == start);

  if (new_size > cur_size){
    idv.ReAllocDistArray(last, new_size);

    assert(idv.GetDistArray(last)[cur_size - 1] == INF8);
    for (size_t pos = cur_size - 1; pos < new_size; pos++){
      idv.GetDistArray(last)[pos] = 0;
    }
    idv.GetDistArray(last)[new_size-1] = INF8;
  }
  idv.GetDistArray(last)[new_size - 2] += count;
}
