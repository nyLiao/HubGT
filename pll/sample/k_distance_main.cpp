#include <cstring>
#include <cstdlib>
#include <iostream>
#include "top_k_pruned_landmark_labeling.hpp"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " (K) (index_file) (output_file)" << endl;
    exit(EXIT_FAILURE);
  }
  
  TopKPrunedLandmarkLabeling kpll;
  
  if (!kpll.LoadIndex(argv[2])) {
    cerr << "error: Load failed" << endl;
    exit(EXIT_FAILURE);
  }
  size_t K = 0, L = strlen(argv[1]);
  for (size_t i = 0; i < L; ++i)
    K = K * 10 + argv[1][i] - 48;
  
  FILE* f = freopen(argv[3], "w", stdout);
  for (int u, v; cin >> u >> v; ) {
    vector<int> dist;
    kpll.KDistanceQuery(u, v, dist);
    
    size_t dlen = dist.size();
    cout << u << " " << v;
    for (size_t i = 0; i < dlen; i++){
      cout << " " << dist[i];
    }
    for (size_t i = dlen; i < K; i++){
      cout << " " << dist[i];
    }
    cout << endl;
  }
  fclose(f);
  exit(EXIT_SUCCESS);
}
