import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from math import factorial, sqrt

a1, a2 = sp.symbols('a1 a2')


# Works only for a state in two channels.
def get_state_coeffs(state, max_power, symbols=(a1, a2), operators_form='unapplied'):
    symb1, symb2 = symbols
    state_coeffs = np.zeros((max_power, max_power), dtype=complex)
    # Cases of only one element in the series
    if state.is_Symbol:
        if state is symb1:
            state_coeffs[1, 0] = 1
        if state is symb2:
            state_coeffs[0, 1] = 1
    elif state.is_Pow:
        if symb1 in state.free_symbols:
            state_coeffs[state.args[1], 0] = 1
        if symb2 in state.free_symbols:
            state_coeffs[0, state.args[1]] = 1
    elif state.is_Mul:
        pows = [0] * 2
        for arg in list(state.args):
            if arg is symb1:
                pows[0] = 1
            elif symb1 in arg.free_symbols:
                pows[0] = arg.args[1]
            if arg is symb2:
                pows[1] = 1
            elif symb2 in arg.free_symbols:
                pows[1] = arg.args[1]
        state_coeffs[pows[0], pows[1]] = complex(list(state.args)[0])
    # Several elements in the sum
    elif state.is_Add:
        for arg in state.args:
            if arg.is_complex:
                state_coeffs[0, 0] = state_coeffs[0, 0] + complex(arg)
            if not arg.is_Mul and not arg.is_complex:
                print('Coeffs. Type is neither mul nor complex. Type: ', type(arg), arg)
            pows = [0] * 2
            for x in list(arg.args):
                if symb1 in x.free_symbols:
                    if x == symb1:
                        pows[0] = 1
                    else:
                        pows[0] = x.args[1]
                if symb2 in x.free_symbols:
                    if x == symb2:
                        pows[1] = 1
                    else:
                        pows[1] = x.args[1]

            if pows[0] + pows[1] > 0:
                state_coeffs[pows[0], pows[1]] = state_coeffs[pows[0], pows[1]] + complex(sp.Poly(arg, domain='CC').coeffs()[0])
    else:
        raise ValueError('Incorrect state type!')

    if operators_form == 'unapplied':
        return state_coeffs
    elif operators_form == 'applied':
        for i in range(len(state_coeffs)):
            for j in range(len(state_coeffs)):
                state_coeffs[i, j] = state_coeffs[i, j] * sqrt(factorial(i) * factorial(j))
        return state_coeffs
    else:
        raise ValueError('Incorrect operators_form type!')


def plot_state(
        state_coeffs,
        name,
        size=8,
        value='abs',
        xlabel='a2 degree',
        ylabel='a1 degree'):
    if value is 'abs':
        plt.matshow(np.abs(state_coeffs)[0:size, 0:size])
        plt.title(name + ', Abs value')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()
        return
    if value is 'real':
        plt.matshow(np.real(state_coeffs)[0:size, 0:size])
        plt.title(name + ', Real value')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()
        return
    if value is 'imag':
        plt.matshow(np.imag(state_coeffs)[0:size, 0:size])
        plt.title(name + ', Imag value')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()
        return
    else:
        raise ValueError('Wrong configuration')


# For input and auxiliary states
def get_state_norm(state):
    norm = 0
    for i in range(len(state)):
        norm = norm + state[i] * np.conj(state[i]) * factorial(i)
    return norm


def diagonal_factorials(len):
    return np.identity(len) * np.array([sqrt(factorial(x)) for x in range(len)])
