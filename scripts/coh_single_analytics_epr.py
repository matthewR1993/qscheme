import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from math import sqrt


def gamma_1(t1, t2, phi):
    return 1j * t1 * np.exp(1j * phi) * sqrt(1 - t2**2) + sqrt(1 - t1**2) * t2


def gamma_2(t1, t2, phi):
    return t1 * t2 * np.exp(1j * phi) - sqrt(1 - t1**2) * sqrt(1 - t2**2)


def gamma_3(t1, t2, phi):
    return t1 * t2 - sqrt(1 - t1**2) * sqrt(1 - t2**2) * np.exp(1j * phi)


def gamma_4(t1, t2, phi):
    return 1j * t1 * sqrt(1 - t2**2) + 1j * t2 * sqrt(1 - t1**2) * np.exp(1j * phi)


# A constant C with the wave.
def c_sl(alph, g1, g2):
    return np.exp(0.5 * np.abs(alph)**2 * (np.abs(g1)**2 + np.abs(g2)**2 - 1))


alpha = 1

bet1 = alpha * g1
bet2 = alpha * g2







