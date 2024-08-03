# cython: language_level=3
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint8_t
from libcpp cimport bool


cdef extern from "ppll.h":
    cdef cppclass PrunedLandmarkLabeling:
        PrunedLandmarkLabeling() except+

        void SetArgs(bool)
        float ConstructIndex(vector[uint32_t] &, vector[uint32_t] &)
        bool LoadIndex(char *)
        bool StoreIndex(char *)

        int QueryDistanceParallel(vector[uint32_t] &, vector[uint32_t] &, vector[int] &)
        int Label(int, vector[int] &, vector[int] &)
        int SNeighbor(int, int, vector[int] &, vector[int] &)
