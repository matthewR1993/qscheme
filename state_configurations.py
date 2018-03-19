# this file consist of different state configurations
import numpy as np
import matplotlib.pyplot as plt


def single_photon(series_length=100):
    state = np.zeros(series_length)
    state[0] = 1
    return state


def plot_state(state):
    plt.plot(state, 'b.')
    pass
