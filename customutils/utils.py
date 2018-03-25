import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


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

    return state_coeffs
