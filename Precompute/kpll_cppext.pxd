# cython: language_level=3
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint8_t
from libcpp cimport bool


cdef extern from "ppll.h":
    cdef cppclass PrunedLandmarkLabeling:
        PrunedLandmarkLabeling() except+

        void SetArgs(bool, int, int, int, int, int)
        void ConstructGraph(vector[uint32_t] &, vector[uint32_t] &, vector[uint32_t] &)
        float ConstructIndex()
        bool LoadIndex(char *)
        bool StoreIndex(char *)

        int QueryDistance(int, int)
        int QueryDistanceParallel(vector[int] &, vector[int] &, vector[int] &)
        int FetchNode(int, vector[int] &, vector[int] &)
        int FetchParallel(vector[int] &, vector[int] &, vector[int] &)
