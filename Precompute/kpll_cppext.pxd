# cython: language_level=3
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint8_t
from libcpp cimport bool


cdef extern from "kpll.cpp":
    pass

cdef extern from "kpll.hpp":
    cdef cppclass TopKPrunedLandmarkLabeling:
        TopKPrunedLandmarkLabeling() except+

        bool ConstructIndex(vector[uint32_t] &, vector[uint32_t] &, uint8_t, bool)
        int KDistanceQuery(int, int, uint8_t, vector[int] &);
        int Label(int, vector[int] &, vector[int] &)
        int SNeighbor(int, int, vector[int] &, vector[int] &)
        int SPush(int, int, float, vector[int] &, vector[float] &)

        double IndexingTime()
        double LoopCountTime()
