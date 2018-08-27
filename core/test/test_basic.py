import numpy as np
import cmath
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
    bs_vals = [(0, 1), (sqrt(0.5), sqrt(0.5)), (sqrt(0.73), sqrt(0.27)), (1, 0)]

    for val in bs_vals:
        t1, r1 = val[0], val[1]
        t2, r2 = val[1], val[0]
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
    with pytest.raises(ValueError):
        det_probability([], 'Invalid option')

    series_length = 3
    state = np.zeros((series_length,) * 4, dtype=complex)
    state[0, 0, 0, 0] = 1
    state[0, 1, 1, 0] = 0.3
    state[2, 1, 1, 0] = 0.4
    state[1, 0, 2, 0] = 0.5
    state[1, 1, 1, 2] = 0.7

    assert det_probability(state, 'FIRST') == 1
    assert det_probability(state, 'THIRD') == 0.91
    assert det_probability(state, 'NONE') == 0
    assert cmath.isclose(det_probability(state, 'BOTH'), 1 - (0.4**2 + 0.5**2 + 0.7**2), rel_tol=1e-8)


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


def test_state_norm_opt():
    series_length = 4
    state1 = np.zeros((series_length,) * 4, dtype=complex)
    assert state_norm_opt(state1) == 0

    state2 = np.zeros((series_length,) * 4, dtype=complex)
    state2[0, 0, 0, 2] = 2
    assert state_norm_opt(state2) == sqrt(2**2 * factorial(2))

    state3 = np.zeros((series_length,) * 4, dtype=complex)
    state3[0, 0, 0, 3] = 3
    state3[0, 2, 0, 0] = 5
    state3[1, 0, 1, 0] = 7
    assert state_norm_opt(state3) == sqrt(3**2 * factorial(3) + 5**2 * factorial(2) + 7**2)


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


def test_dens_matrix_with_trace_opt():
    with pytest.raises(ValueError):
        dens_matrix_with_trace_opt(np.array([1, 2, 3]), np.array([1, 2]))

    size = 3

    state1 = np.zeros((size,) * 4, dtype=complex)
    rho_expected1 = np.zeros((size,) * 4, dtype=complex)
    rho1 = dens_matrix_with_trace_opt(state1, state1)
    assert_array_equal(rho1, rho_expected1)

    state2 = np.zeros((size,) * 4, dtype=complex)
    state2[1, 0, 1, 0] = 1
    rho_expected2 = np.zeros((size,) * 4, dtype=complex)
    rho_expected2[0, 0, 0, 0] = 1
    rho2 = dens_matrix_with_trace_opt(state2, state2)
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
    rho3 = dens_matrix_with_trace_opt(state3, state3)
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
    bs_vals = [(0, 1), (sqrt(0.5), sqrt(0.5)), (sqrt(0.73), sqrt(0.27)), (1, 0)]

    for val in bs_vals:
        t1, r1 = val[0], val[1]
        rho_input1 = np.zeros((size,) * 4, dtype=complex)
        rho_out1 = bs_densmatrix_transform(rho_input1, t1, r1)
        rho_out_expected1 = np.zeros((size*2,) * 4, dtype=complex)
        assert_array_equal(rho_out1, rho_out_expected1)

        t2, r2 = val[0], val[1]
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
    size = 3

    rho1 = np.zeros((size,) * 2, dtype=complex)
    assert cmath.isclose(log_entropy(rho1), 0, rel_tol=1e-9)

    rho2 = np.zeros((size,) * 2, dtype=complex)
    rho2[0, 0] = 0.5
    rho2[1, 1] = 0.35
    rho2[2, 2] = 0.15
    assert cmath.isclose(log_entropy(rho2), -(0.5 * np.log2(0.5) + 0.35 * np.log2(0.35) + 0.15 * np.log2(0.15)), rel_tol=1e-9)


def test_partial_transpose():
    size = 4

    rho1 = np.zeros((size,) * 4, dtype=complex)
    assert_array_equal(rho1, partial_transpose(rho1))

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[1, 1, 1, 0] = 3
    rho2[1, 1, 0, 0] = 1
    rho2[0, 1, 0, 2] = 5
    rho2[2, 1, 3, 3] = 4
    rho_ptr2 = partial_transpose(rho2)
    rho_ptr_expected2 = np.zeros((size,) * 4, dtype=complex)
    rho_ptr_expected2[1, 0, 1, 1] = 3
    rho_ptr_expected2[1, 0, 0, 1] = 1
    rho_ptr_expected2[0, 2, 0, 1] = 5
    rho_ptr_expected2[2, 3, 3, 1] = 4
    assert_array_equal(rho_ptr2, rho_ptr_expected2)


def test_linear_entropy():
    size = 3

    rho1 = np.zeros((size,) * 2, dtype=complex)
    assert linear_entropy(rho1) == 1

    rho2 = np.zeros((size,) * 2, dtype=complex)
    rho2[0, 0] = 1
    rho2[1, 1] = 3
    rho2[1, 2] = 2
    rho2[2, 2] = 7
    assert linear_entropy(rho2) == -58


def test_reorganise_dens_matrix():
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    rho_reorg1 = reorganise_dens_matrix(rho1)
    rho_reorg_expect1 = np.zeros((size**2,) * 2, dtype=complex)
    assert_array_equal(rho_reorg1, rho_reorg_expect1)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[0, 0, 0, 0] = 1
    rho2[1, 1, 0, 0] = 3
    rho2[1, 0, 2, 1] = 7
    rho2[2, 0, 2, 0] = 5
    rho_reorg2 = reorganise_dens_matrix(rho2)
    rho_reorg_expect2 = np.zeros((size**2,) * 2, dtype=complex)
    rho_reorg_expect2[0, 0] = 1
    rho_reorg_expect2[4, 0] = 3
    rho_reorg_expect2[3, 7] = 7
    rho_reorg_expect2[6, 6] = 5
    assert_array_equal(rho_reorg2, rho_reorg_expect2)


def test_negativity():
    with pytest.raises(ValueError):
        negativity([], neg_type=None)
    size = 3

    rho1 = np.zeros((size,) * 4, dtype=complex)
    cmath.isclose(negativity(rho1, neg_type='raw'), 0, rel_tol=1e-9)
    cmath.isclose(negativity(rho1, neg_type='logarithmic'), 0, rel_tol=1e-9)

    a02, a11, a20 = 7 + 1j, 3, 5 - 3j
    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[0, 2, 0, 2] = a02 * np.conj(a02)
    rho2[0, 2, 1, 1] = a02 * np.conj(a11)
    rho2[0, 2, 2, 0] = a02 * np.conj(a20)
    rho2[1, 1, 0, 2] = a11 * np.conj(a02)
    rho2[1, 1, 1, 1] = a11 * np.conj(a11)
    rho2[1, 1, 2, 0] = a11 * np.conj(a20)
    rho2[2, 0, 0, 2] = a20 * np.conj(a02)
    rho2[2, 0, 1, 1] = a20 * np.conj(a11)
    rho2[2, 0, 2, 0] = a20 * np.conj(a20)
    neg_expected = (abs(a02) + abs(a20)) * abs(a11) + abs(a02) * abs(a20)
    assert cmath.isclose(negativity(rho2, neg_type='raw'), neg_expected, rel_tol=1e-9)
    assert cmath.isclose(negativity(rho2, neg_type='logarithmic'), np.log2(2 * neg_expected + 1), rel_tol=1e-9)


def test_phase_modulation():
    size = 3
    phase = 0.43 * np.pi

    rho1 = np.zeros((size,) * 4, dtype=complex)
    rho_mod1 = phase_modulation(rho1, phase)
    assert_array_equal(rho1, rho_mod1)

    rho2 = np.zeros((size,) * 4, dtype=complex)
    rho2[1, 0, 1, 0] = 1
    rho2[1, 0, 2, 2] = 3
    rho2[1, 1, 2, 2] = 7
    rho2[0, 2, 2, 0] = 2
    rho_mod2 = phase_modulation(rho2, 0)
    assert_array_equal(rho2, rho_mod2)

    rho_mod3 = phase_modulation(rho2, phase)
    rho_mod_excepted3 = np.zeros((size,) * 4, dtype=complex)
    rho_mod_excepted3[1, 0, 1, 0] = 1
    rho_mod_excepted3[1, 0, 2, 2] = 3 * np.exp(-1j * phase * 2)
    rho_mod_excepted3[1, 1, 2, 2] = 7 * np.exp(-1j * phase)
    rho_mod_excepted3[0, 2, 2, 0] = 2 * np.exp(1j * phase * 2)
    assert_array_equal(rho_mod3, rho_mod_excepted3)


def test_phase_modulation_state():
    size = 3
    phase = 0.47 * np.pi

    state1 = np.zeros((size,) * 2, dtype=complex)
    state_mod1 = phase_modulation_state(state1, phase)
    assert_array_equal(state_mod1, state1)

    state2 = np.zeros((size,) * 2, dtype=complex)
    state2[1, 0] = 5
    state2[2, 1] = 3
    state_mod2 = phase_modulation_state(state2, 0)
    assert_array_equal(state_mod2, state2)

    state3 = np.zeros((size,) * 2, dtype=complex)
    state3[1, 0] = 7
    state3[2, 1] = 11
    state_mod3 = phase_modulation_state(state3, phase)
    state_mod_expected3 = np.zeros((size,) * 2, dtype=complex)
    state_mod_expected3[1, 0] = 7 * np.exp(1j * phase)
    state_mod_expected3[2, 1] = 11 * np.exp(1j * phase * 2)
    assert_array_equal(state_mod3, state_mod_expected3)


def test_make_state_appliable():
    size = 4

    state1 = np.zeros((size,) * 2, dtype=complex)
    state_appl1 = make_state_appliable(state1)
    assert_array_equal(state_appl1, state1)

    state2 = np.zeros((size,) * 2, dtype=complex)
    state2[1, 0] = 1
    state2[1, 3] = 3
    state2[0, 2] = 5
    state2[3, 2] = 7
    state_appl2 = make_state_appliable(state2)
    state_appl_expected2 = np.zeros((size,) * 2, dtype=complex)
    state_appl_expected2[1, 0] = 1
    state_appl_expected2[1, 3] = 3 * sqrt(6)
    state_appl_expected2[0, 2] = 5 * sqrt(2)
    state_appl_expected2[3, 2] = 7 * sqrt(12)
    assert_array_equal(state_appl2, state_appl_expected2)


def make_state_appliable_4ch():
    size = 4

    state1 = np.zeros((size,) * 4, dtype=complex)
    state_appl1 = make_state_appliable_4ch(state1)
    assert_array_equal(state_appl1, state1)

    state2 = np.zeros((size,) * 4, dtype=complex)
    state2[1, 1, 1, 1] = 3
    state2[1, 2, 1, 2] = 5
    state2[3, 2, 1, 2] = 7
    state_appl2 = make_state_appliable_4ch(state2)
    state_appl_expected2 = np.zeros((size,) * 4, dtype=complex)
    state_appl_expected2[1, 1, 1, 1] = 3
    state_appl_expected2[1, 2, 1, 2] = 5 * sqrt(4)
    state_appl_expected2[3, 2, 1, 2] = 7 * sqrt(24)
    assert_array_equal(state_appl2, state_appl_expected2)


def test_bs_parameters():
    T_min = 0.35
    T_max = 0.73

    assert_array_equal(bs_parameters(T_min, T_max, 1), (np.array([sqrt(0.35)]), np.array([sqrt(1 - 0.35)])))
    assert_array_equal(bs_parameters(T_min, T_max, 2), (np.array([sqrt(0.35), sqrt(0.73)]), np.array([sqrt(1 - 0.35), sqrt(1 - 0.73)])))
    assert_array_equal(bs_parameters(T_min, T_max, 3), (np.array([sqrt(0.35), sqrt(0.54), sqrt(0.73)]), np.array([sqrt(1 - 0.35), sqrt(1 - 0.54), sqrt(1 - 0.73)])))
