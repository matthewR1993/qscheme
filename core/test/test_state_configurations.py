import pytest
from numpy.testing import assert_array_equal

from ..state_configurations import *


def test_single_photon_configuration():
    with pytest.raises(ValueError):
        single_photon(0)
    with pytest.raises(ValueError):
        single_photon(1)
    assert_array_equal(single_photon(3), np.array([0, 1, 0]))
    assert_array_equal(single_photon(2), np.array([0, 1]))


def test_fock_state_configuration():
    with pytest.raises(ValueError):
        fock_state(0, series_length=0)
    with pytest.raises(IndexError):
        fock_state(1, series_length=1)
    assert_array_equal(fock_state(0, series_length=1), np.array([1]))
    assert_array_equal(fock_state(1, series_length=2), np.array([0, 1]))
    assert_array_equal(fock_state(2, series_length=5), np.array([0, 0, 1/sqrt(factorial(2)), 0, 0]))
    assert_array_equal(fock_state(4, series_length=5), np.array([0, 0, 0, 0, 1/sqrt(factorial(4))]))
    assert_array_equal(fock_state(0, series_length=5), np.array([1, 0, 0, 0, 0]))


def test_coherent_state_configuration():
    with pytest.raises(ValueError):
        coherent_state(series_length=0, alpha=1)
    assert_array_equal(coherent_state(series_length=3, alpha=0), np.array([1, 0, 0]))
    assert_array_equal(coherent_state(series_length=1, alpha=1), np.array([exp(-abs(1)**2 / 2)]))
    assert_array_equal(coherent_state(series_length=1, alpha=5), np.array([exp(-abs(5) ** 2 / 2)]))
    alpha = 7
    assert_array_equal(coherent_state(series_length=3, alpha=alpha), np.array([exp(-abs(alpha)**2 / 2), exp(-abs(alpha)**2 / 2) * alpha, exp(-abs(alpha)**2 / 2) * alpha**2 / factorial(2)]))


def test_squeezed_vacuum_configuration():
    with pytest.raises(ValueError):
        squeezed_vacuum(series_length=0, squeezing_amp=1, squeezing_phase=0)
    with pytest.raises(ValueError):
        squeezed_vacuum(series_length=3, squeezing_amp=1, squeezing_phase=0)
    squeezing_amp,  squeezing_phase = 0.7, 0.3
    assert_array_equal(squeezed_vacuum(series_length=2, squeezing_amp=squeezing_amp, squeezing_phase=squeezing_phase), np.array([1 / sqrt(np.cosh(squeezing_amp)), 0]))


def test_squeezed_coherent_state_configuration():
    with pytest.raises(ValueError):
        squeezed_coherent_state(series_length=0, alpha=1, squeezing_amp=1, squeezing_phase=0)
    squeezing_amp, alpha, squeezing_phase = 0.87, 5, 0.3
    result = squeezed_coherent_state(series_length=1, alpha=alpha, squeezing_amp=squeezing_amp, squeezing_phase=squeezing_phase)
    const = (1 / sqrt(np.cosh(squeezing_amp))) * cm.exp(- 0.5 * abs(alpha)**2 - 0.5 * np.conj(alpha)**2 * cm.exp(1j * squeezing_phase) * np.tanh(squeezing_amp))
    gamma = alpha * np.cosh(squeezing_amp) + np.conj(alpha) * cm.exp(1j * squeezing_phase) * np.sinh(squeezing_amp)
    herm_coeff_arr = np.array([1])
    expected = np.array([const * herm.hermval((gamma / cm.sqrt(cm.exp(1j * squeezing_phase) * np.sinh(2 * squeezing_amp))), herm_coeff_arr)])
    assert_array_equal(result, expected)
