from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t, uint8_t
import numpy as np
cimport numpy as np

from kpll_cppext cimport TopKPrunedLandmarkLabeling


cdef class PyPLL:
    cdef TopKPrunedLandmarkLabeling c_pll

    def __cinit__(self):
        self.c_pll = TopKPrunedLandmarkLabeling()

    def construct_index(self, np.ndarray[uint32_t, ndim=2] edge_index, unsigned int K, bool directed):
        ns, nt = edge_index
        self.c_pll.ConstructIndex(ns, nt, K, directed)
        return self.c_pll.LoopCountTime() + self.c_pll.IndexingTime()

    def k_distance_query(self, int s, int t, unsigned int K):
        cdef vector[int] result
        self.c_pll.KDistanceQuery(s, t, K, result)

        return result

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

    def s_push(self, int v, int size, float alpha):
        cdef vector[int] nodes
        cdef vector[float] dist
        length = self.c_pll.SPush(v, size, alpha, nodes, dist)
        return nodes, dist, length
