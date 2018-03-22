# this file consist of different state configurations
import numpy as np


DEF_SERIES_LEN = 100


def single_photon(series_length=DEF_SERIES_LEN):
    state = np.zeros(series_length)
    state[0] = 1
    return state
