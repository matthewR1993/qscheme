import numpy as np
from math import factorial

cimport numpy as cnp
cimport cython


cdef int max_size = 30
cdef fact_arr_ = np.array([factorial(x) for x in range(max_size)], dtype=np.complex)
cdef fact_sqrt_arr_ = np.sqrt(fact_arr_)
cdef complex[:] fact_arr = fact_arr_
cdef complex[:] fact_sqrt_arr = fact_sqrt_arr_


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bs_matrix_transform_copt(cnp.ndarray[cnp.complex_t, ndim=4, mode='c'] rho_in, double t, double r):
    cdef int sz = len(rho_in)
    cdef cnp.ndarray[cnp.complex_t, ndim=4, mode='c'] rho_out = np.zeros((sz*2,) * 4, dtype=np.complex)

    # Create memory view to efficiently access arrays.
    cdef complex[:, :, :, :] mv_rho_out = rho_out
    cdef complex[:, :, :, :] mv_rho_in = rho_in

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

                                    mv_rho_out[d1, d2, d1_, d2_] = mv_rho_out[d1, d2, d1_, d2_] + mv_rho_in[p1, p2, p1_, p2_] * coeff1 * coeff2

    return mv_rho_out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef two_bs2x4_transform_copt(double t1, double r1, double t2, double r2, cnp.ndarray[cnp.complex_t, ndim=2, mode='c'] state_in):
    cdef int sz = len(state_in)
    cdef cnp.ndarray[cnp.complex_t, ndim=4, mode='c'] state_out = np.zeros((sz,) * 4, dtype=np.complex)
    cdef cnp.complex128_t tc1 = t1
    cdef cnp.complex128_t rc1 = r1
    cdef cnp.complex128_t tc2 = t2
    cdef cnp.complex128_t rc2 = r2

    # Create memory view to efficiently access arrays.
    cdef complex[:, :, :, :] mv_state_out = state_out
    cdef complex[:, :] mv_state_in = state_in

    for m in range(sz):
        for n in range(sz):
            for k in range(m + 1):
                for l in range(n + 1):
                    coeff = mv_state_in[m, n] * pow(tc1, m - k) * pow(1j * rc1, k) * pow(tc2, (n - l)) * pow(1j * rc2, l) * fact_arr[m] * fact_arr[n] / (fact_arr[k] * fact_arr[m - k] * fact_arr[l] * fact_arr[n - l])
                    mv_state_out[k, m - k, l, n - l] = mv_state_out[k, m - k, l, n - l] + coeff

    return np.mv_state_out
