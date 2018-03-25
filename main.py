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
input_series_length = 8
auxiliary_series_length = 8
max_power = input_series_length + auxiliary_series_length


# set up input state as a Taylor series
# input_st[n] = state with 'n' photons !!!
input_st = single_photon(2)
# input_st = coherent_state(input_series_length, alpha=1)

# set up auxiliary state as a Taylor series
# auxiliary_st = single_photon(2)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)

# Setting up state before first BS.

# Method with symbolic expressions
a1, a2 = sp.symbols('a1 a2')
g = 0
for i in range(len(input_st)):
    g = g + input_st[i]*(a1**(i))
f = 0
for i in range(len(auxiliary_st)):
    f = f + auxiliary_st[i]*(a2**(i))

# state = g(a1) * f(a2)
state1 = g * f

# State after mixing at first BS
state2 = state1
b1, b2 = sp.symbols('b1 b2')

state2 = state2.subs(a1, (t1*b1 + 1j*t1*b2))
state2 = state2.subs(a2, (t1*b2 + 1j*t1*b1))

# put 'a' operators back
state2 = state2.subs(b1, a1)
state2 = state2.subs(b2, a2)

state2 = sp.expand(state2)
print('State 2:', state2)

# Plot state2
state2_coeffs = get_state_coeffs(state2, max_power + 1)

plt.matshow(np.abs(state2_coeffs)[0:8, 0:8])
plt.title('State2, Abs value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

plt.matshow(np.real(state2_coeffs)[0:8, 0:8])
plt.title('State2, Real value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

plt.matshow(np.imag(state2_coeffs)[0:8, 0:8])
plt.title('State2, Imag value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

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

# Consider ideal detectors first
# Both detector were clicked - Det1 and Det2.

# state4 is a state after measurement
state_4pre = 0

# iterate over all sum members
for arg in state3.args:
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
        state_4pre = state_4pre + arg * sqrt(factorial(b1_power)) * sqrt(factorial(b3_power)) / (b1**b1_power * b3**b3_power)

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

state5_coef = np.zeros((max_power, max_power), dtype=complex)

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


coef_abs = np.abs(state5_coef)
coef_real = np.real(state5_coef)
coef_imag = np.imag(state5_coef)


# Abs
plt.matshow(coef_abs[0:8, 0:8])
plt.title('Output state, Abs value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

# Real
plt.matshow(coef_real[0:8, 0:8])
plt.title('Output state, Real value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()

# Imag
plt.matshow(coef_imag[0:8, 0:8])
plt.title('Output state, Imag value')
plt.xlabel('a2')
plt.ylabel('a1')
plt.colorbar()
plt.show()


# plot input states
#plt.bar(list(range(len(input_st))), input_st, width=1, edgecolor='c')
plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], width=1, edgecolor='c')
plt.title('Input state')
plt.xlabel('Number of photons')
plt.show()

plt.bar(list(range(len(auxiliary_st))), auxiliary_st, color='g', width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], color='g', width=1, edgecolor='c')
plt.title('Auxiliary state')
plt.xlabel('Number of photons')
plt.show()

# save setup configuration in file TODO
