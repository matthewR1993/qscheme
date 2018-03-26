import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from math import sqrt, factorial


def get_state_coeffs(state, max_power):
    a1, a2 = sp.symbols('a1 a2')
    state_coeffs = np.zeros((max_power, max_power), dtype=complex)
    for arg in state.args:
        pows = [0] * 2
        for x in list(arg.args):
            if a1 in x.free_symbols:
                if x == a1:
                    pows[0] = 1
                else:
                    pows[0] = x.args[1]
            if a2 in x.free_symbols:
                if x == a2:
                    pows[1] = 1
                else:
                    pows[1] = x.args[1]

        if pows[0] + pows[1] > 0:
            # print('powers: ', pows[0], pows[1])
            state_coeffs[pows[0], pows[1]] = complex(sp.Poly(arg, domain='CC').coeffs()[0])
        else:
            state_coeffs[0, 0] = arg

    return state_coeffs


def state_after_measurement(state_before):
    b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')
    state_after_pre = 0

    # iterate over all sum members
    for arg in state_before.args:
        if b1 in arg.free_symbols:
            # finding power of b1
            for item in arg.args:
                if item == b1:
                    b1_power = 1
                elif item.is_Pow and b1 in item.free_symbols:
                    b1_power = item.args[1]
            # finding power of b3
            b3_power = 0
            if b3 in arg.free_symbols:
                for item in arg.args:
                    if item == b3:
                        b3_power = 1
                    elif item.is_Pow and b3 in item.free_symbols:
                        b3_power = item.args[1]
            state_4pre = state_after_pre + arg * sqrt(factorial(b1_power)) * sqrt(factorial(b3_power)) / (
                        b1 ** b1_power * b3 ** b3_power)

    state_after = 0
    # filter constants
    for arg in state_after_pre.args:
        if b2 in list(arg.free_symbols) or b4 in list(arg.free_symbols):
            state_after = state_after + arg

    return state_after
