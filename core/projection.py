import numpy as np
import sympy as sp
from math import sqrt, factorial


def measure_state(state_before, clicked):
    if clicked is 'BOTH':
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
                state_after_pre = state_after_pre + arg * sqrt(factorial(b1_power)) * sqrt(factorial(b3_power)) / (
                            b1 ** b1_power * b3 ** b3_power)

        state_after = 0
        # filter constants ??? TODO
        for arg in state_after_pre.args:
            if b2 in list(arg.free_symbols) or b4 in list(arg.free_symbols):
                state_after = state_after + arg
        return state_after
    # TODO
    if clicked is 'FIRST':
        return
    else:
        raise ValueError('Wrong configuration')
