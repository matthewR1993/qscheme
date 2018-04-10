import numpy as np
import sympy as sp
from math import sqrt, factorial


def measure_state(state_before, clicked):
    # Both detectors were clicked
    b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')
    if clicked is 'BOTH':
        state_after_pre = 0

        for arg in state_before.args:
            if not arg.is_Mul and not arg.is_complex:
                print('Det. Type is neither mul nor complex. Type : ', type(arg), arg)
            # Condition for both detectors being clicked
            if b1 in arg.free_symbols and b3 in arg.free_symbols:
                # Finding constant beta
                beta = complex(arg.args[0])
                # Finding power of b1
                for item in arg.args:
                    if item == b1:
                        b1_power = 1
                    elif item.is_Pow and b1 in item.free_symbols:
                        b1_power = item.args[1]
                # Finding power of b3
                b3_power = 0
                if b3 in arg.free_symbols:
                    for item in arg.args:
                        if item == b3:
                            b3_power = 1
                        elif item.is_Pow and b3 in item.free_symbols:
                            b3_power = item.args[1]
                # Finding power of b2
                b2_power = 0
                if b2 in arg.free_symbols:
                    for item in arg.args:
                        if item == b2:
                            b2_power = 1
                        elif item.is_Pow and b2 in item.free_symbols:
                            b2_power = item.args[1]
                # Finding power of b4
                b4_power = 0
                if b4 in arg.free_symbols:
                    for item in arg.args:
                        if item == b4:
                            b4_power = 1
                        elif item.is_Pow and b4 in item.free_symbols:
                            b4_power = item.args[1]
                state_after_pre = state_after_pre + arg * sqrt(factorial(b1_power) * factorial(b3_power)) / (
                            b1 ** b1_power * b3 ** b3_power)
        state_after = state_after_pre
        return state_after
    # Only first detector was clicked
    if clicked is 'FIRST':
        state_after_pre = 0
        for arg in state_before.args:
            if not arg.is_Mul and not arg.is_complex:
                print('Det. Type is neither mul nor complex. Type : ', type(arg), arg)
            # Condition for only first detector being clicked
            if b1 in arg.free_symbols and b3 not in arg.free_symbols:
                # Finding constant beta
                beta = complex(arg.args[0])
                # Finding power of b1
                for item in arg.args:
                    if item == b1:
                        b1_power = 1
                    elif item.is_Pow and b1 in item.free_symbols:
                        b1_power = item.args[1]
                # Finding power of b2
                b2_power = 0
                if b2 in arg.free_symbols:
                    for item in arg.args:
                        if item == b2:
                            b2_power = 1
                        elif item.is_Pow and b2 in item.free_symbols:
                            b2_power = item.args[1]
                # Finding power of b4
                b4_power = 0
                if b4 in arg.free_symbols:
                    for item in arg.args:
                        if item == b4:
                            b4_power = 1
                        elif item.is_Pow and b4 in item.free_symbols:
                            b4_power = item.args[1]
                state_after_pre = state_after_pre + 0
                state_after_pre = state_after_pre + arg * sqrt(factorial(b1_power)) / (
                            b1 ** b1_power)
        state_after = state_after_pre
        return state_after
    if clicked is 'NONE':
        state_after_pre = 0

        for arg in state_before.args:
            if not arg.is_Mul and not arg.is_complex:
                print('Det. Type is neither mul nor complex. Type : ', type(arg), arg)
            # None of detectors were clicked
            if b1 not in arg.free_symbols and b3 not in arg.free_symbols:
                state_after_pre = state_after_pre + arg
        return state_after_pre
    else:
        raise ValueError('Wrong configuration')
