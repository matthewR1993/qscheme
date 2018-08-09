import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from ..state_configurations import single_photon, fock_state
from ..basic import *


def test_bs2x2_transform():
    series_length = 2
    state1 = np.tensordot(single_photon(series_length), single_photon(series_length), axes=0)
    t1, r1 = 1, 0
    assert_array_equal(bs2x2_transform(t1, r1, state1), np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    t2, r2 = 0, 1
    assert_array_equal(bs2x2_transform(t2, r2, state1), np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]]))
    t3, r3 = sqrt(0.5), sqrt(0.5)
    assert_allclose(bs2x2_transform(t3, r3, state1), np.array([[0, 0, 0.5j], [0, 0, 0], [0.5j, 0, 0]]))
    t4, r4 = sqrt(0.73), sqrt(0.27)
    assert_allclose(bs2x2_transform(t4, r4, state1), np.array([[0, 0, 1j*t4*r4], [0, t4**2 - r4**2, 0], [1j*t4*r4, 0, 0]]))


def test_two_bs2x4_transform():
    series_length = 3
    t1, r1 = sqrt(0.57), sqrt(0.43)
    t2, r2 = sqrt(0.17), sqrt(0.83)

    # Two single photons.
    state1 = np.tensordot(single_photon(series_length), single_photon(series_length), axes=0)
    out_state1 = np.zeros((series_length,) * 4, dtype=complex)
    out_state1[0, 1, 0, 1] = t1 * t2
    out_state1[0, 1, 1, 0] = 1j * t1 * r2
    out_state1[1, 0, 0, 1] = 1j * r1 * t2
    out_state1[1, 0, 1, 0] = - r1 * r2
    assert_array_equal(two_bs2x4_transform(t1, r1, t2, r2, state1), out_state1)

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
    assert_allclose(two_bs2x4_transform(t1, r1, t2, r2, state2), out_state2)


def test_detection():
    with pytest.raises(ValueError):
        detection([], 'Invalid option')
    series_length = 3
    t1, r1 = sqrt(0.57), sqrt(0.43)
    t2, r2 = sqrt(0.17), sqrt(0.83)

    # Two single photons after 2BS.
    input_state = np.zeros((series_length,) * 4, dtype=complex)
    input_state[0, 2, 0, 2] = t1**2 * t2**2
    input_state[0, 2, 1, 1] = 2j * t1**2 * t2 * r2
    input_state[0, 2, 2, 0] = - t1**2 * r2**2
    input_state[1, 1, 0, 2] = 2j * t1 * r1 * t2**2
    input_state[1, 1, 1, 1] = - 4 * t1 * r1 * t2 * r2
    input_state[1, 1, 2, 0] = - 2j * t1 * r1 * r2**2
    input_state[2, 0, 0, 2] = - r1**2 * t2**2
    input_state[2, 0, 1, 1] = - 2j * r1**2 * t2 * r2
    input_state[2, 0, 2, 0] = r1**2 * r2**2

    output1 = detection(input_state, 'BOTH')
    output_expected1 = np.array(input_state)
    output_expected1[0, 2, 0, 2] = 0
    output_expected1[0, 2, 1, 1] = 0
    output_expected1[0, 2, 2, 0] = 0
    output_expected1[1, 1, 0, 2] = 0
    output_expected1[2, 0, 0, 2] = 0
    assert_array_equal(output1, output_expected1)

    output2 = detection(input_state, 'NONE')
    output_expected2 = np.zeros((series_length,) * 4, dtype=complex)
    output_expected2[0, 2, 0, 2] = t1 ** 2 * t2 ** 2
    assert_array_equal(output2, output_expected2)

    output3 = detection(input_state, 'FIRST')
    output_expected3 = np.zeros((series_length,) * 4, dtype=complex)
    output_expected3[1, 1, 0, 2] = 2j * t1 * r1 * t2**2
    output_expected3[2, 0, 0, 2] = - r1 ** 2 * t2 ** 2
    assert_array_equal(output3, output_expected3)

    output4 = detection(input_state, 'THIRD')
    output_expected4 = np.zeros((series_length,) * 4, dtype=complex)
    output_expected4[0, 2, 1, 1] = 2j * t1**2 * t2 * r2
    output_expected4[0, 2, 2, 0] = - t1**2 * r2**2
    assert_array_equal(output4, output_expected4)


def test_det_probability():
    pass


def test_state_norm():
    series_length = 4
    state1 = np.zeros((series_length,) * 4, dtype=complex)
    assert state_norm(state1) == 0

    state2 = np.zeros((series_length,) * 4, dtype=complex)
    state2[0, 0, 0, 2] = 2
    assert state_norm(state2) == sqrt(2**2 * factorial(2))

    state3 = np.zeros((series_length,) * 4, dtype=complex)
    state3[0, 0, 0, 3] = 3
    state3[0, 2, 0, 0] = 5
    state3[1, 0, 1, 0] = 7
    assert state_norm(state3) == sqrt(3**2 * factorial(3) + 5**2 * factorial(2) + 7**2)


def test_dens_matrix_with_trace():
    with pytest.raises(ValueError):
        dens_matrix_with_trace(np.array([1, 2, 3]), np.array([1, 2]))

    size = 3

    state1 = np.zeros((size,) * 4, dtype=complex)
    rho_expected1 = np.zeros((size,) * 4, dtype=complex)
    rho1 = dens_matrix_with_trace(state1, state1)
    assert_array_equal(rho1, rho_expected1)

    state2 = np.zeros((size,) * 4, dtype=complex)
    state2[1, 0, 1, 0] = 1
    rho_expected2 = np.zeros((size,) * 4, dtype=complex)
    rho_expected2[0, 0, 0, 0] = 1
    rho2 = dens_matrix_with_trace(state2, state2)
    assert_array_equal(rho2, rho_expected2)

    state3 = np.zeros((size,) * 4, dtype=complex)
    state3[1, 1, 2, 0] = 3
    state3[1, 2, 2, 1] = 5 + 3j
    state3[1, 1, 1, 0] = 7 - 1j
    rho_expected3 = np.zeros((size,) * 4, dtype=complex)
    rho_expected3[1, 0, 1, 0] = 18 + 50
    rho_expected3[1, 0, 2, 1] = 6 * sqrt(2) * (5 - 3j)
    rho_expected3[2, 1, 1, 0] = 6 * sqrt(2) * (5 + 3j)
    rho_expected3[2, 1, 2, 1] = 4 * 34
    rho3 = dens_matrix_with_trace(state3, state3)
    assert_array_equal(rho3, rho_expected3)


def test_dens_matrix():
    size = 4

    state1 = np.zeros((size,) * 2, dtype=complex)
    rho_expected1 = np.zeros((size,) * 4, dtype=complex)
    rho1 = dens_matrix(state1)
    assert_array_equal(rho1, rho_expected1)

    state2 = np.zeros((size,) * 2, dtype=complex)
    state2[1, 1] = 5 + 3j
    state2[0, 3] = 7j
    state2[0, 0] = 3
    rho_expected2 = np.zeros((size,) * 4, dtype=complex)
    rho_expected2[0, 0, 0, 0] = 9
    rho_expected2[0, 0, 0, 3] = - 21j
    rho_expected2[0, 0, 1, 1] = 15 - 9j
    rho_expected2[0, 3, 0, 0] = 21j
    rho_expected2[0, 3, 1, 1] = 35j + 21
    rho_expected2[0, 3, 0, 3] = 49
    rho_expected2[1, 1, 0, 0] = 15 + 9j
    rho_expected2[1, 1, 0, 3] = - 35j + 21
    rho_expected2[1, 1, 1, 1] = 34
    rho2 = dens_matrix(state2)
    assert_array_equal(rho2, rho_expected2)


def test_dens_matrix_4ch():
    size = 3

    state1 = np.zeros((size,) * 4, dtype=complex)
    rho1 = dens_matrix_4ch(state1)
    rho_expected1 = np.zeros((size,) * 8, dtype=complex)
    assert_array_equal(rho1, rho_expected1)

    state2 = np.zeros((size,) * 4, dtype=complex)
    state2[0, 0, 0, 0] = 3
    state2[1, 0, 2, 0] = 5 + 7j
    rho2 = dens_matrix_4ch(state2)
    rho_expected2 = np.zeros((size,) * 8, dtype=complex)
    rho_expected2[0, 0, 0, 0, 0, 0, 0, 0] = 9
    rho_expected2[0, 0, 0, 0, 1, 0, 2, 0] = 15 - 21j
    rho_expected2[1, 0, 2, 0, 0, 0, 0, 0] = 15 + 21j
    rho_expected2[1, 0, 2, 0, 1, 0, 2, 0] = 74
    assert_array_equal(rho2, rho_expected2)


def test_trace_channel():
    with pytest.raises(ValueError):
        trace_channel([], channel=7)

    size = 3

    rho_input1 = np.zeros((size,) * 4, dtype=complex)
    rho_out1 = trace_channel(rho_input1, channel=4)
    rho_out_expected1 = np.zeros((size,) * 2, dtype=complex)
    assert_array_equal(rho_out1, rho_out_expected1)

    rho_input2 = np.zeros((size,) * 4, dtype=complex)
    rho_out2 = trace_channel(rho_input2, channel=2)
    rho_out_expected2 = np.zeros((size,) * 2, dtype=complex)
    assert_array_equal(rho_out2, rho_out_expected2)

    rho_input3 = np.zeros((size,) * 4, dtype=complex)
    rho_input3[0, 0, 0, 0] = 3
    rho_input3[1, 0, 1, 0] = 5 + 7j
    rho_input3[1, 2, 1, 1] = 1 - 3j
    rho_input3[0, 1, 0, 2] = 5 + 3j
    rho_out3 = trace_channel(rho_input3, channel=2)
    rho_out4 = trace_channel(rho_input3, channel=4)
    rho_out_expected3 = np.zeros((size,) * 2, dtype=complex)
    rho_out_expected3[0, 0] = 8 + 7j
    rho_out_expected3[2, 1] = 1 - 3j
    rho_out_expected3[1, 2] = 5 + 3j
    assert_array_equal(rho_out3, rho_out_expected3)
    rho_out_expected4 = np.zeros((size,) * 2, dtype=complex)
    rho_out_expected4[0, 0] = 3
    rho_out_expected4[1, 1] = 5 + 7j
    assert_array_equal(rho_out4, rho_out_expected4)


def test_bs_densmatrix_transform():
    size = 4

    t1, r1 = sqrt(0.73), sqrt(0.27)
    rho_input1 = np.zeros((size,) * 4, dtype=complex)
    rho_out1 = bs_densmatrix_transform(rho_input1, t1, r1)
    rho_out_expected1 = np.zeros((size*2,) * 4, dtype=complex)
    assert_array_equal(rho_out1, rho_out_expected1)

    t2, r2 = sqrt(0.73), sqrt(0.27)
    rho_input2 = np.zeros((size,) * 4, dtype=complex)
    rho_input2[0, 0, 0, 0] = 1
    rho_input2[2, 0, 0, 1] = 3 + 5j
    rho_input2[1, 0, 2, 0] = 1 - 7j
    rho_out2 = bs_densmatrix_transform(rho_input2, t2, r2)
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


def test_prob_distr():
    size = 3

    rho_input1 = np.zeros((size,) * 4, dtype=complex)
    prob1 = prob_distr(rho_input1)
    prob_expected1 = np.zeros((size,) * 2, dtype=complex)
    assert_array_equal(prob1, prob_expected1)

    rho_input2 = np.zeros((size,) * 4, dtype=complex)
    rho_input2[1, 1, 1, 1] = 1
    rho_input2[1, 2, 1, 2] = 3
    rho_input2[0, 2, 1, 0] = 3 + 7j
    prob2 = prob_distr(rho_input2)
    prob_expected2 = np.zeros((size,) * 2, dtype=complex)
    prob_expected2[1, 1] = 1
    prob_expected2[1, 2] = 3
    assert_array_equal(prob2, prob_expected2)


def test_log_entropy():
    pass


def test_partial_transpose():
    pass


def test_linear_entropy():
    pass


def test_reorganise_dens_matrix():
    pass


def test_negativity():
    pass


def test_phase_modulation():
    pass


def test_phase_modulation_state():
    pass


def test_make_state_appliable():
    pass


def make_state_appliable_4ch():
    pass
