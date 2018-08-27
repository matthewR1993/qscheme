from numpy.testing import assert_array_equal, assert_allclose
from math import sqrt
import numpy as np

from ..optimized.transformations import bs_matrix_transform_opt


def test_bs_matrix_transform_opt():
    size = 4

    # [(t, r),]
    bs_vals = [(0, 1), (sqrt(0.5), sqrt(0.5)), (sqrt(0.73), sqrt(0.27)), (1, 0)]

    for val in bs_vals:
        t1, r1 = val[0], val[1]
        rho_input1 = np.zeros((size,) * 4, dtype=complex)
        rho_out1 = bs_matrix_transform_opt(rho_input1, t1, r1)
        rho_out_expected1 = np.zeros((size*2,) * 4, dtype=complex)
        assert_array_equal(rho_out1, rho_out_expected1)

        t2, r2 = val[0], val[1]
        rho_input2 = np.zeros((size,) * 4, dtype=complex)
        rho_input2[0, 0, 0, 0] = 1
        rho_input2[2, 0, 0, 1] = 3 + 5j
        rho_input2[1, 0, 2, 0] = 1 - 7j
        rho_out2 = bs_matrix_transform_opt(rho_input2, t2, r2)
        rho_out_expected2 = np.zeros((size*2,) * 4, dtype=complex)
        rho_out_expected2[0, 0, 0, 0] = 1
        rho_out_expected2[2, 0, 0, 1] = (3 + 5j) * t2**3
        rho_out_expected2[2, 0, 1, 0] = - (3 + 5j) * 1j * t2**2 * r2
        rho_out_expected2[1, 1, 0, 1] = (3 + 5j) / sqrt(2) * 2j * t2**2 * r2
        rho_out_expected2[1, 1, 1, 0] = (3 + 5j) / sqrt(2) * 2 * t2 * r2**2
        rho_out_expected2[0, 2, 0, 1] = - (3 + 5j) * t2 * r2**2
        rho_out_expected2[0, 2, 1, 0] = (3 + 5j) * 1j * r2**3
        rho_out_expected2[1, 0, 2, 0] = (1 - 7j) * t2**3
        rho_out_expected2[1, 0, 1, 1] = - (1 - 7j) / sqrt(2) * 2j * t2**2 * r2
        rho_out_expected2[1, 0, 0, 2] = - (1 - 7j) / sqrt(2) * t2 * r2**2 * sqrt(2)
        rho_out_expected2[0, 1, 2, 0] = (1 - 7j) * 1j * t2**2 * r2
        rho_out_expected2[0, 1, 1, 1] = (1 - 7j) / sqrt(2) * 2 * t2 * r2**2
        rho_out_expected2[0, 1, 0, 2] = - (1 - 7j) * 1j * r2**3
        assert_allclose(rho_out2, rho_out_expected2, rtol=1e-07, atol=1e-09,)
