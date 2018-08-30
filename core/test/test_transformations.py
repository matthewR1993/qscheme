from numpy.testing import assert_array_equal, assert_allclose
from math import sqrt
import numpy as np

from ..state_configurations import single_photon, fock_state
from ..optimized.transformations import bs_matrix_transform_copt, two_bs2x4_transform_copt


def test_bs_matrix_transform_copt():
    size = 4

    bs_vals = [(0, 1), (sqrt(0.5), sqrt(0.5)), (sqrt(0.73), sqrt(0.27)), (1, 0)]

    for val in bs_vals:
        t1, r1 = val[0], val[1]
        rho_input1 = np.zeros((size,) * 4, dtype=complex)
        rho_out1 = bs_matrix_transform_copt(rho_input1, t1, r1)
        rho_out_expected1 = np.zeros((size*2,) * 4, dtype=complex)
        assert_array_equal(rho_out1, rho_out_expected1)

        t2, r2 = val[0], val[1]
        rho_input2 = np.zeros((size,) * 4, dtype=complex)
        rho_input2[0, 0, 0, 0] = 1
        rho_input2[2, 0, 0, 1] = 3 + 5j
        rho_input2[1, 0, 2, 0] = 1 - 7j
        rho_out2 = bs_matrix_transform_copt(rho_input2, t2, r2)
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


def test_two_bs2x4_transform_copt():
    series_length = 5
    bs_vals = [(0, 1), (sqrt(0.5), sqrt(0.5)), (sqrt(0.73), sqrt(0.27)), (1, 0)]

    for val in bs_vals:
        t1, r1 = val[0], val[1]
        t2, r2 = val[1], val[0]
        # Two single photons.
        state1 = np.tensordot(single_photon(series_length), single_photon(series_length), axes=0)
        state1 = np.array(state1, dtype=complex)
        out_state1 = np.zeros((series_length,) * 4, dtype=complex)
        out_state1[0, 1, 0, 1] = t1 * t2
        out_state1[0, 1, 1, 0] = 1j * t1 * r2
        out_state1[1, 0, 0, 1] = 1j * r1 * t2
        out_state1[1, 0, 1, 0] = - r1 * r2
        print(state1)
        st1 = two_bs2x4_transform_copt(t1, r1, t2, r2, state1)
        print(st1)
        assert_array_equal(two_bs2x4_transform_copt(t1, r1, t2, r2, state1), out_state1)

        # Two Fock states with n=2.
        state2 = np.tensordot(fock_state(2, series_length), fock_state(2, series_length), axes=0)
        out_state2 = np.zeros((series_length,) * 4, dtype=complex)
        out_state2[0, 2, 0, 2] = t1**2 * t2**2
        out_state2[0, 2, 1, 1] = 2j * t1**2 * t2 * r2
        out_state2[0, 2, 2, 0] = - t1**2 * r2**2
        out_state2[1, 1, 0, 2] = 2j * t1 * r1 * t2**2
        out_state2[1, 1, 1, 1] = - 4 * t1 * r1 * t2 * r2
        out_state2[1, 1, 2, 0] = - 2j * t1 * r1 * r2**2
        out_state2[2, 0, 0, 2] = - r1**2 * t2**2
        out_state2[2, 0, 1, 1] = - 2j * r1**2 * t2 * r2
        out_state2[2, 0, 2, 0] = r1**2 * r2**2
        out_state2 = out_state2 * 0.5  # initial state is unapplied
        assert_allclose(two_bs2x4_transform_copt(t1, r1, t2, r2, state2), out_state2)
