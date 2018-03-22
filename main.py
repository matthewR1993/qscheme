import numpy as np
import qutip as qt
from math import sqrt, factorial
import matplotlib.pyplot as plt
from customutils.utils import *
from state_configurations import single_photon, coherent_state
import sympy as sp


# Scheme has only two channels in first area
# four in the middle and two in the end.

# Set up default scheme parameters.
# These r and t are from 0 to 1. Beam splitter with absorption: t^2 + r^2 + a^2 = 1
a1 = 0
t1 = sqrt(0.5)
r1 = sqrt(1 - pow(t1, 2) - pow(a1, 2))

a2 = 0
t2 = sqrt(0.5)
r2 = sqrt(1 - pow(t2, 2) - pow(a2, 2))

a3 = 0
t3 = sqrt(0.5)
r3 = sqrt(1 - pow(t3, 2) - pow(a3, 2))

a4 = 0
t4 = sqrt(0.5)
r4 = sqrt(1 - pow(t4, 2) - pow(a4, 2))

# can be set small for simple configurations
series_length = 7


# set up input state as a Taylor series
# input_st = single_photon(series_length)
input_st = coherent_state(series_length, 1)
# plot_state(input_st)

# set up auxiliary state as a Taylor series
auxiliary_st = single_photon(2)


# Setting up state before first BS.
# state1 = np.empty((series_length, series_length), dtype=object)

# for i in range(len(auxiliary_st)):
#     for j in range(len(auxiliary_st)):
#         state1[i, j] = [auxiliary_st[i], input_st[j]]

# Method with symbolic expressions
a1, a2 = sp.symbols('a1 a2')
g = 0
for i in range(len(input_st)):
    g = g + input_st[i]*(a1**(i+1))
f = 0
for i in range(len(auxiliary_st)):
    f = f + auxiliary_st[i]*(a2**(i+1))

# state = g(a1) * f(a2)
state1 = g * f

# State after mixing at first BS
state2 = state1
b1, b2 = sp.symbols('b1 b2')

state2 = state2.subs(a1, (t1*b1 + 1j*t1*b2))
state2 = state2.subs(a2, (t1*b2 + 1j*t1*b1))

# put 'a' back
state2 = state2.subs(b1, a1)
state2 = state2.subs(b2, a2)

print('State 2:', sp.expand(state2))

# 'state2' is a state after BS

# a1 goes to 2nd BS with t2, r2 and split into b1 and b2. Therefore: a1 -> t2*b1 + 1j*t2*b2
# a2 goes to 3rd BS with t3, r3 and split into b3 and b4. Therefore: a2 -> t3*b3 + 1j*t3*b4
# state3 is a state after these two BSs
state3 = state2
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')

state3 = state3.subs(a1, (t2*b1 + 1j*t2*b2))
state3 = state3.subs(a2, (t3*b3 + 1j*t3*b4))

state3 = sp.expand(state3)

print('State 3:', state3)

# Calculation of all probabilities of detector to catch all possible numbers of photons TODO

# Consider ideal detectors first
# First detector clicked - Det1.

# state4 is a state after measurement
state_4pre = 0

# iterate over all sum members
for arg in state3.args:
    if b1 in arg.free_symbols:
        # finding power of b1
        for item in arg.args:
            if item == b1:
                b1_power = 1
            elif item.is_Pow:
                b1_power = item.args[1]
        state_4pre = state_4pre + sqrt(factorial(b1_power)) * arg / sp.Pow(b1, b1_power)

state4 = 0
# filter constants
for arg in state_4pre.args:
    if b2 in list(arg.free_symbols) or b4 in list(arg.free_symbols):
        state4 = state4 + arg

print('State 4:', state4)

# Now mixing state in a  fourth BS
# Final state is state5
# b2 -> t4*a1 + 1j*t4*a2
# b4 -> t4*a2 + 1j*t4*a1
state5 = state4

state5 = state5.subs(b2, (t4*a1 + 1j*t4*a2))
state5 = state5.subs(b4, (t4*a2 + 1j*t4*a1))

state5 = sp.expand(state5)

print('State 5:', state5)

# Plotting final state.
# Matrix of coefficients.

state5_coef = np.zeros((2*series_length, 2*series_length), dtype=complex)

for arg in state5.args:
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
        state5_coef[pows[0], pows[1]] = complex(sp.Poly(arg, domain='CC').coeffs()[0])


coef_abs = np.absolute(state5_coef)
coef_real = np.real(state5_coef)
coef_imag = np.imag(state5_coef)


# Abs
plt.matshow(coef_abs)
plt.title('Abs value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

# Real
plt.matshow(coef_real)
plt.title('Real value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

# Imag
plt.matshow(coef_imag)
plt.title('Imag value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()
