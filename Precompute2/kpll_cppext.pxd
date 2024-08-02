# cython: language_level=3
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint8_t
from libcpp cimport bool


cdef extern from "ppll.h":
    cdef cppclass PrunedLandmarkLabeling:
        PrunedLandmarkLabeling() except+

        bool ConstructIndex(vector[int] &, vector[int] &)
