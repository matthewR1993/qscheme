import numpy as np
import qutip as qt
from math import sqrt
import matplotlib.pyplot as plt
from customutils import *
from state_configurations import single_photon, plot_state
import sympy as sp

# Scheme has only two channels in first area
# four in the middle and two in the end.

# Set up default scheme parameters.
# These r and t are from 0 to 1. Ideal beam splitter: t^2 + r^2 = 1
t1 = sqrt(0.5)
r1 = sqrt(1 - pow(t1, 2))

t2 = sqrt(0.5)
r2 = sqrt(1 - pow(t2, 2))

t3 = sqrt(0.5)
r3 = sqrt(1 - pow(t3, 2))

t4 = sqrt(0.5)
r4 = sqrt(1 - pow(t4, 2))

# can be set small for simple configurations
series_length = 20


# set up input state as a Taylor series
input_st = single_photon(series_length)
# plot_state(input_st)

# set up auxiliary state as a Taylor series
auxiliary_st = single_photon(series_length)


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

# 'state2' is a state after BS

# a1 goes to 2nd BS with t2, r2 and split into b1 and b2. Therefore: a1 -> t2*b1 + 1j*t2*b2
# a2 goes to 3rd BS with t3, r3 and split into b3 and b4. Therefore: a2 -> t3*b3 + 1j*t3*b4
# state3 is a state after these two BSs
state3 = state2
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')

state3 = state3.subs(a1, (t2*b1 + 1j*t2*b2))
state3 = state3.subs(a2, (t3*b3 + 1j*t3*b4))

# Calculation of all probabilities of detector to catch all possible numbers of photons

# Max number of photons that might happen to go to channel with detector
# lets set as 1, however need to be calculated
photon_num = 1


sp.expand(state3)




