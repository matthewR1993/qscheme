# This submodule consist of functions for different state configurations.
# All state configurations are with not applied operators!!!
import numpy as np
from math import factorial, exp, sqrt
import cmath as cm
import numpy.polynomial.hermite as herm


DEF_SERIES_LEN = 100


def single_photon(series_length=DEF_SERIES_LEN):
    '''
    Generating single photon state in a Fock basis.
    :param series_length: Lenght of the state.
    :return: Single photon state as an array.
    '''
    if series_length < 2:
        raise ValueError('The series length should be >= 2')
    state = np.zeros(series_length)
    state[1] = 1
    return state


def fock_state(n, series_length=DEF_SERIES_LEN):
    '''
    Generating a Fock state.
    :param n: Fock number
    :param series_length: Lenght of the state.
    :return: General Fock state state as an array.
    '''
    if series_length < 1:
        raise ValueError('The series length should be a positive integer')
    state = np.zeros(series_length)
    state[n] = 1 / sqrt(factorial(n))
    return state


def coherent_state(series_length=DEF_SERIES_LEN, alpha=1):
    '''
    Generating a coherent state in a Fock basis.
    :param series_length: Lenght of the state.
    :param alpha: Alpha
    :return: Coherent state as an array.
    '''
    if series_length < 1:
        raise ValueError('The series length should be a positive integer')
    state = np.zeros(series_length, dtype=np.complex128)
    for n in range(series_length):
        state[n] = exp(-abs(alpha)**2 / 2) * alpha**n / factorial(n)
    return state


def squeezed_vacuum(series_length=DEF_SERIES_LEN, squeezing_amp=1, squeezing_phase=0):
    '''
    Generating a squezed vacuum state in a Fock basis.
    :param series_length: Lenght of the state.
    :param squeezing_amp: Squeezing parameter amplitude.
    :param squeezing_phase: Squeezing parameter phase.
    :return: Squezed vacuum state as an array.
    '''
    if series_length % 2 != 0 or series_length < 1:
        raise ValueError('The series length should be positive even integer')
    state = np.zeros(series_length, dtype=np.complex128)
    for n in range(int(series_length/2)):
        m = 2 * n
        state[m] = 1/sqrt(factorial(m)) * (1 / sqrt(np.cosh(squeezing_amp))) * ((-1)**n) * (sqrt(factorial(2 * n))/((2**n) * factorial(n))) * cm.exp(1j * n * squeezing_phase) * (np.tanh(squeezing_amp)**n)
    return state


def squeezed_coherent_state(series_length=DEF_SERIES_LEN, alpha=1, squeezing_amp=1, squeezing_phase=0):
    '''
    Generating a squezed coherent state in a Fock basis.
    :param series_length: Lenght of the state.
    :param alpha: Coheret parameter alpha.
    :param squeezing_amp: Squeezing parameter amplitude.
    :param squeezing_phase: Squeezing parameter phase.
    :return: Squezed coherent state as an array.
    '''
    if series_length < 1:
        raise ValueError('The series length should be a positive integer')
    state = np.zeros(series_length, dtype=np.complex128)
    const = (1 / sqrt(np.cosh(squeezing_amp))) * cm.exp(- 0.5 * abs(alpha)**2 - 0.5 * np.conj(alpha)**2 * cm.exp(1j * squeezing_phase) * np.tanh(squeezing_amp))
    for n in range(series_length):
        herm_coeff_arr = np.zeros(series_length)
        herm_coeff_arr[n] = 1
        gamma = alpha * np.cosh(squeezing_amp) + np.conj(alpha) * cm.exp(1j * squeezing_phase) * np.sinh(squeezing_amp)
        state[n] = 1/sqrt(factorial(n)) * const * (0.5 * cm.exp(1j*squeezing_phase) * np.tanh(squeezing_amp)) ** (n/2) / cm.sqrt(factorial(n)) * herm.hermval((gamma / cm.sqrt(cm.exp(1j * squeezing_phase) * np.sinh(2 * squeezing_amp))), herm_coeff_arr)
    return state
