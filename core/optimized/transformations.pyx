# Cython version of dens_matrix_with_trace function

import numpy as np

cimport numpy as np
cimport cython


cdef extern from "math.h":
    double sqrt(double m)


def factorial(int x):
    cdef int m = x
    cdef int i

    if x <= 1:
        return 1
    else:
        for i in range(1, x):
            m = m * i
        return m


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def dm_with_trace(np.ndarray[np.complex_t, ndim=4] l_vect, np.ndarray[np.complex_t, ndim=4] r_vect):
    cdef int sz = len(l_vect)
    cdef np.ndarray[np.complex_t, ndim=4] dm = np.zeros([sz, sz, sz, sz], dtype=np.complex)
    r_vect_conj = np.conj(r_vect)

    cdef unsigned int p2, p2_, p4, p4_, k1, k3

    cdef complex matrix_sum

    for p2 in range(sz):
        for p2_ in range(sz):
            for p4 in range(sz):
                for p4_ in range(sz):
                    matrix_sum = 0
                    for k1 in range(sz):
                        for k3 in range(sz):
                            matrix_sum += l_vect[k1, p2, k3, p4] * r_vect_conj[k1, p2_, k3, p4_] * factorial(k1) * factorial(k3) * sqrt(factorial(p2) * factorial(p4) * factorial(p2_) * factorial(p4_))
                    dm[p2, p4, p2_, p4_] = matrix_sum
    return dm
