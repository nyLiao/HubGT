from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t, uint8_t
import numpy as np
cimport numpy as np

from kpll_cppext cimport PrunedLandmarkLabeling


cdef class PyPLL:
    cdef PrunedLandmarkLabeling c_pll

    def __cinit__(self):
        self.c_pll = PrunedLandmarkLabeling()

    def construct_index(self, np.ndarray[uint32_t, ndim=2] edge_index, unsigned int K, bool directed, bool quiet):
        ns, nt = edge_index
        self.c_pll.SetArgs(quiet)
        return self.c_pll.ConstructIndex(ns, nt)
