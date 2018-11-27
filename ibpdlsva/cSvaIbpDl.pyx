# distutils: language = c++
# distutils: sources = [ibpdlsva/sources/sva_algorithm.cpp, ibpdlsva/sources/sva_inpainting.cpp, ibpdlsva/sources/containers.cpp, ibpdlsva/sources/utils.cpp, ibpdlsva/sources/cythonInterface.cpp]

# Cython and Eigen
# https://github.com/wouterboomsma/eigency

from eigency.core cimport *
from libcpp cimport bool

cdef extern from "cythonInterface.h":
    cdef cppclass _SvaDl "SvaDl_cythonInterface":
        _SvaDl "SvaDl_cythonInterface"() except +
        MatrixXd &get_dic()
        MatrixXd &get_coefs()
        void run(FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor] &, int, double, double)

    cdef cppclass _SvaDlInpainting "SvaDlInpainting_cythonInterface":
        _SvaDlInpainting "SvaDlInpainting_cythonInterface"() except +
        MatrixXd &get_dic()
        MatrixXd &get_coefs()
        void run(FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor] &, FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor] &, int, double, double)


# This will be exposed to Python
cdef class CSvaDl:
    cdef _SvaDl *thisptr;

    def __cinit__(self):
        self.thisptr = new _SvaDl()

    def __dealloc__(self):
        del self.thisptr

    def run(self, np.ndarray[np.float64_t, ndim=2] arrayY, int nbIt, double lbd1, double lbd2):
        self.thisptr.run(FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor](arrayY), nbIt, lbd1, lbd2)

    def get_dic(self):
        return ndarray(self.thisptr.get_dic())

    def get_coefs(self):
        return ndarray(self.thisptr.get_coefs())


# This will also be exposed to Python
cdef class CSvaDlInpainting:
    cdef _SvaDlInpainting *thisptr;

    def __cinit__(self):
        self.thisptr = new _SvaDlInpainting()

    def __dealloc__(self):
        del self.thisptr

    def run(self, np.ndarray[np.float64_t, ndim=2] arrayY, np.ndarray[int, ndim=2] arrayMask, int nbIt, double lbd1, double lbd2):
        self.thisptr.run(FlattenedMapWithOrder[Array, double, Dynamic, Dynamic, RowMajor](arrayY), FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor](arrayMask), nbIt, lbd1, lbd2)

    def get_dic(self):
        return ndarray(self.thisptr.get_dic())

    def get_coefs(self):
        return ndarray(self.thisptr.get_coefs())

