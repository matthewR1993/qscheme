import matplotlib.pyplot as plt
import numpy as np
from math import factorial, sqrt


# For input and auxiliary states in single channel
# Takes state that is not applied
def get_state_norm(state):
    norm = 0
    for i in range(len(state)):
        norm = norm + state[i] * np.conj(state[i]) * factorial(i)
    return norm


# Takes unapplied state
def get_state_norm_2ch(state):
    norm = 0
    state_conj = np.conj(state)
    for p1 in range(len(state)):
        for p2 in range(len(state)):
            norm = norm + state[p1, p2] * state_conj[p1, p2] * factorial(p1) * factorial(p2)
    return norm


def diagonal_factorials(len):
    return np.identity(len) * np.array([sqrt(factorial(x)) for x in range(len)])
