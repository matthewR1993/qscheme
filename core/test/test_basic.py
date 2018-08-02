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
    pass


def test_dens_matrix(state):
    pass
