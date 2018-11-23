import numpy as np
from math import factorial, sqrt


def get_state_norm(state):
    """
    Calculate the norm of the state.
    :param state: Single channel(mode) unapplied state.
    :return: Norm of the state.
    """
    norm = 0
    for i in range(len(state)):
        norm = norm + state[i] * np.conj(state[i]) * factorial(i)
    return norm


# Takes unapplied state
def get_state_norm_2ch(state):
    """
    Calculate the norm of the state.
    :param state: Two channels(modes) unapplied state.
    :return: Norm of the state.
    """
    norm = 0
    state_conj = np.conj(state)
    for p1 in range(len(state)):
        for p2 in range(len(state)):
            norm = norm + state[p1, p2] * state_conj[p1, p2] * factorial(p1) * factorial(p2)
    return norm


def diagonal_factorials(l):
    """
    Diagonal matrix of factorials.
    :param l: Matrix size.
    :return: Diagonal matrix of factorials
    """
    return np.identity(l) * np.array([sqrt(factorial(x)) for x in range(l)])
