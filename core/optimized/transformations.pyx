import numpy as np
from math import factorial

cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def bs_matrix_transform_opt(cnp.ndarray[cnp.complex_t, ndim=4] rho_in, double t, double r):
    cdef int max_size = 30
    cdef fact_arr = np.array([factorial(x) for x in range(max_size)], dtype=np.complex)
    cdef fact_sqrt_arr = np.sqrt(fact_arr)

    cdef int sz = len(rho_in)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] rho_out = np.zeros([sz*2, sz*2, sz*2, sz*2], dtype=np.complex)

    cdef int d1, d2, d1_, d2_
    cdef cnp.complex128_t coeff1, coeff2
    cdef cnp.complex128_t tc = t
    cdef cnp.complex128_t rc = r

    for p1 in range(sz):
        for p1_ in range(sz):
            for p2 in range(sz):
                for p2_ in range(sz):

                    for n in range(p1 + 1):
                        for k in range(p2 + 1):
                            for n_ in range(p1_ + 1):
                                for k_ in range(p2_ + 1):
                                    d1 = p1 - n + k
                                    d2 = n + p2 - k
                                    coeff1 = pow(tc, p1 - n + p2 - k) * pow(1j * rc, n + k) * fact_sqrt_arr[d1] * fact_sqrt_arr[d2] * fact_sqrt_arr[p1] * fact_sqrt_arr[p2] / (fact_arr[n] * fact_arr[p1 - n] * fact_arr[k] * fact_arr[p2 - k])

                                    d1_ = p1_ - n_ + k_
                                    d2_ = n_ + p2_ - k_
                                    coeff2 = pow(tc, p1_ - n_ + p2_ - k_) * pow(-1j * rc, n_ + k_) * fact_sqrt_arr[d1_] * fact_sqrt_arr[d2_] * fact_sqrt_arr[p1_] * fact_sqrt_arr[p2_] / (fact_arr[n_] * fact_arr[p1_ - n_] * fact_arr[k_] * fact_arr[p2_ - k_])

                                    rho_out[d1, d2, d1_, d2_] = rho_out[d1, d2, d1_, d2_] + rho_in[p1, p2, p1_, p2_] * coeff1 * coeff2

    return rho_out


cdef int sz = 7
cdef int max_size = 30

cdef fact_arr = np.array([factorial(x) for x in range(max_size)], dtype=np.complex)
cdef fact_sqrt_arr = np.sqrt(fact_arr)



cdef extern from "transformations_cource.c":
    void bs_matrix_transform(complex<double> (&input_matrix)[7][7][7][7], complex<double> (&output_matrix)[14][14][14][14], double t, double r)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bs_matrix_transform_opt2(cnp.ndarray[cnp.complex_t, ndim=4, mode='c'] rho_in, double t, double r):
    cdef cnp.complex128_t rho_out[14][14][14][14]

    bs_matrix_transform(rho_in, rho_out)

    return 0
