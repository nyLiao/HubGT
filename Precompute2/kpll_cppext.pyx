from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t, uint8_t
import os
import numpy as np
cimport numpy as np

from kpll_cppext cimport PrunedLandmarkLabeling


cdef class PyPLL:
    cdef PrunedLandmarkLabeling c_pll

    def __cinit__(self):
        self.c_pll = PrunedLandmarkLabeling()

    def get_index(self, np.ndarray[uint32_t, ndim=2] edge_index, np.ndarray[uint32_t, ndim=1] alias_inv, str path_cache, bool quiet, bool index):
        path_cache = path_cache.replace('/log', '/cache')
        os.makedirs(path_cache, exist_ok=True)
        path_cache += '/index.bin'
        if not os.path.exists(path_cache) or index:
            ns, nt = edge_index
            self.c_pll.SetArgs(quiet)
            self.c_pll.ConstructGraph(ns, nt, alias_inv)
            res = self.c_pll.ConstructIndex()
            if not quiet:
                self.store_index(path_cache)
            return res
        else:
            self.c_pll.SetArgs(quiet)
            return 1.0 - self.load_index(path_cache)

    def construct_index(self, np.ndarray[uint32_t, ndim=2] edge_index, np.ndarray[uint32_t, ndim=1] alias_inv, bool quiet):
        ns, nt = edge_index
        self.c_pll.SetArgs(quiet)
        self.c_pll.ConstructGraph(ns, nt, alias_inv)
        return self.c_pll.ConstructIndex()

    def load_index(self, str filename):
        return self.c_pll.LoadIndex(filename.encode('utf-8'))

    def store_index(self, str filename):
        return self.c_pll.StoreIndex(filename.encode('utf-8'))

    def k_distance_parallel(self, np.ndarray[int, ndim=1] ns, np.ndarray[int, ndim=1] nt):
        cdef vector[int] result
        result = np.empty_like(ns, dtype=np.int32)
        self.c_pll.QueryDistanceParallel(ns, nt, result)
        return result

    def glabel(self, int v):
        cdef vector[int] nodes
        cdef vector[int] dist
        length = self.c_pll.Global(v, nodes, dist)
        return nodes, dist, length

    def label(self, int v):
        cdef vector[int] nodes
        cdef vector[int] dist
        length = self.c_pll.Label(v, nodes, dist)
        return nodes, dist, length

    def s_neighbor(self, int v, int size):
        cdef vector[int] nodes
        cdef vector[int] dist
        length = self.c_pll.SNeighbor(v, size, nodes, dist)
        return nodes, dist, length
