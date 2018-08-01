from ..state_configurations import *
import numpy as np
import pytest
from numpy.testing import assert_array_equal


def test_single_photon_configuration():
    with pytest.raises(IndexError):
        single_photon(0)
    with pytest.raises(IndexError):
        single_photon(1)
    assert_array_equal(single_photon(3), np.array([0, 1, 0]))
    assert_array_equal(single_photon(2), np.array([0, 1]))


def test_fock_state_configuration():
    with pytest.raises(IndexError):
        fock_state(0, series_length=0)
    with pytest.raises(IndexError):
        fock_state(1, series_length=1)
    assert_array_equal(fock_state(0, series_length=1), np.array([1]))
    assert_array_equal(fock_state(1, series_length=2), np.array([0, 1]))
    assert_array_equal(fock_state(2, series_length=5), np.array([0, 0, 1/sqrt(factorial(2)), 0, 0]))
    assert_array_equal(fock_state(4, series_length=5), np.array([0, 0, 0, 0, 1/sqrt(factorial(4))]))
    assert_array_equal(fock_state(0, series_length=5), np.array([1, 0, 0, 0, 0]))


def test_coherent_state_configuration():
    assert_array_equal(coherent_state(series_length=3, alpha=0), np.array([1, 0, 0]))
    assert_array_equal(coherent_state(series_length=1, alpha=1), np.array([exp(-abs(1)**2 / 2)]))
    assert_array_equal(coherent_state(series_length=1, alpha=5), np.array([exp(-abs(5) ** 2 / 2)]))
    alpha = 7
    assert_array_equal(coherent_state(series_length=3, alpha=alpha), np.array([exp(-abs(alpha)**2 / 2), exp(-abs(alpha)**2 / 2) * alpha, exp(-abs(alpha)**2 / 2) * alpha**2 / factorial(2)]))


def test_squeezed_vacuum_configuration():
    pass
