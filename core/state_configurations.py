# this file consist of different state configurations
import numpy as np
from math import factorial, exp

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


# TODO
def squeezed_vacuum(series_length=DEF_SERIES_LEN, squeezing=1):
    return 0


# TODO
def squeezed_coherent_state(series_length=DEF_SERIES_LEN, squeezing=1, alpha=1):
    return 0
