from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t, uint8_t
import os
import numpy as np
cimport numpy as np

from kpll_cppext cimport PrunedLandmarkLabeling


cdef class PyPLL:
    cdef PrunedLandmarkLabeling c_pll
    cdef public str path_cache
    cdef public bool quiet
    cdef public int seed
    cdef public int n_fetch
    cdef public int n_bp
    cdef public int n_spt
    cdef public int n_inv

    def __getstate__(self):
        return (self.path_cache, self.quiet, self.seed, self.n_fetch, self.n_bp, self.n_spt, self.n_inv)

    def __setstate__(self, state):
        self.path_cache = state[0]
        self.quiet = state[1]
        self.seed = state[2]
        self.n_fetch = state[3]
        self.n_bp = state[4]
        self.n_spt = state[5]
        self.n_inv = state[6]
        self.c_pll = PrunedLandmarkLabeling()
        self.c_pll.SetArgs(self.quiet, self.seed, self.n_fetch, self.n_bp, self.n_spt, self.n_inv)
        try:
            self.load_index(self.path_cache)
        except:
            raise Exception('Failed to load index')

    def get_index(self, np.ndarray[uint32_t, ndim=2] edge_index, np.ndarray[uint32_t, ndim=1] alias_inv, str path_cache, bool index, bool quiet):
        path_cache = path_cache.replace('/log', '/cache')
        os.makedirs(path_cache, exist_ok=True)
        path_cache += '/index.bin'
        self.path_cache = path_cache
        if not os.path.exists(path_cache) or index:
            ns, nt = edge_index
            self.c_pll.ConstructGraph(ns, nt, alias_inv)
            res = self.c_pll.ConstructIndex()
            if not quiet:
                self.store_index(path_cache)
            return res
        else:
            try:
                res = 1.0 - self.load_index(path_cache)
                if res < 0.0:
                    ns, nt = edge_index
                    self.c_pll.ConstructGraph(ns, nt, alias_inv)
                    res = self.c_pll.ConstructIndex()
                    if not quiet:
                        self.store_index(path_cache)
                return res
            except:
                ns, nt = edge_index
                self.c_pll.ConstructGraph(ns, nt, alias_inv)
                res = self.c_pll.ConstructIndex()
                if not quiet:
                    self.store_index(path_cache)
                return res

    def construct_index(self, np.ndarray[uint32_t, ndim=2] edge_index, np.ndarray[uint32_t, ndim=1] alias_inv):
        ns, nt = edge_index
        self.c_pll.ConstructGraph(ns, nt, alias_inv)
        return self.c_pll.ConstructIndex()

    def set_args(self, bool quiet, int seed, int n_fetch, int n_bp, int n_spt, int n_inv):
        self.c_pll = PrunedLandmarkLabeling()
        self.quiet = quiet
        self.seed = seed
        self.n_fetch = n_fetch
        self.n_bp = n_bp
        self.n_spt = n_spt
        self.n_inv = n_inv
        self.c_pll.SetArgs(quiet, seed, n_fetch, n_bp, n_spt, n_inv)

    def load_index(self, str filename):
        return self.c_pll.LoadIndex(filename.encode('utf-8'))

    def store_index(self, str filename):
        return self.c_pll.StoreIndex(filename.encode('utf-8'))

    def k_distance(self, int ns, int nt):
        return self.c_pll.QueryDistance(ns, nt)

    def k_distance_parallel(self, np.ndarray[int, ndim=1] ns, np.ndarray[int, ndim=1] nt):
        cdef vector[int] result
        result = np.empty_like(ns, dtype=np.int32)
        self.c_pll.QueryDistanceParallel(ns, nt, result)
        return result

    def fetch_node(self, int v):
        cdef vector[int] nodes
        cdef vector[int] dist
        length = self.c_pll.FetchNode(v, nodes, dist)
        return nodes, dist, length

    def fetch_parallel(self, np.ndarray[int, ndim=1] ns, int n_fetch):
        cdef vector[int] nodes
        cdef vector[int] dist
        nodes = np.empty(ns.shape[0] * n_fetch, dtype=np.int32)
        dist = np.empty(ns.shape[0] * n_fetch * n_fetch, dtype=np.int32)
        self.c_pll.FetchParallel(ns, nodes, dist)
        return nodes, dist
