# this file consist of different state configurations
import numpy as np
from math import factorial, exp, sqrt

DEF_SERIES_LEN = 100


def single_photon(series_length=DEF_SERIES_LEN):
    state = np.zeros(series_length)
    state[1] = 1
    return state


def coherent_state(series_length=DEF_SERIES_LEN, alpha=1):
    state = np.zeros(series_length)
    for n in range(series_length):
        state[n] = exp(-abs(alpha)**2 / 2) * alpha**n / factorial(n)
    return state


def squeezed_vacuum(series_length=DEF_SERIES_LEN, squeezing_amp=1, squeezing_phase=1):
    if series_length % 2 != 0:
        print('Warning: series length should be even number')
        series_length = series_length + 1
    state = np.zeros(series_length)
    for n in range(int(series_length / 2)):
        state[2*n] = 1 / np.cosh(squeezing_amp) * (-1)**n * sqrt(factorial(2*n))/(2**n * factorial(n)) * np.exp(1j*n*squeezing_phase) * np.tanh(squeezing_amp)**n
    return state


# TODO
def squeezed_coherent_state(series_length=DEF_SERIES_LEN, squeezing=1, alpha=1):
    return 0
