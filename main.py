import sys

from customutils.utils import *
from core.projection import measure_state
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *


# Parameters for states
input_series_length = 10
auxiliary_series_length = 10
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!

# INPUT
# input_st = single_photon(2)
input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(2)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement detectors configuration
DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
# DET_CONF = 'FIRST'  # 1st detector clicked


# Setting up state before first BS.
a1, a2 = sp.symbols('a1 a2')
g = 0
for i in range(len(input_st)):
    g = g + input_st[i]*(a1**i)
f = 0
for i in range(len(auxiliary_st)):
    f = f + auxiliary_st[i]*(a2**i)

# Initial state = g(a1) * f(a2)
state1 = g * f

state1_coeffs = get_state_coeffs(sp.expand(state1), max_power + 1)

plot_state(state1_coeffs, 'Initial State',  size=8, value='real')

# State after mixing at first BS
state2 = state1
b1, b2 = sp.symbols('b1 b2')

# a1 -> t1*a1 + 1j*r1*a2
state2 = state2.subs(a1, (t1*b1 + 1j*r1*b2))
state2 = state2.subs(a2, (t1*b2 + 1j*r1*b1))

# put 'a' operators back
state2 = state2.subs(b1, a1)
state2 = state2.subs(b2, a2)

state2 = sp.expand(state2)
print('State 2:', state2)

# Plot state2
state2_coeffs = get_state_coeffs(state2, max_power + 1)

# plot_state(state2_coeffs, 'State2',  size=8, value='abs')
# plot_state(state2_coeffs, 'State2',  size=8, value='real')
# plot_state(state2_coeffs, 'State2',  size=8, value='imag')

# 'state2' is a state after BS

# a1 goes to 2nd BS with t2, r2 and split into b1 and b2. Therefore: a1 -> t2*b1 + 1j*r2*b2
# a2 goes to 3rd BS with t3, r3 and split into b3 and b4. Therefore: a2 -> t3*b3 + 1j*r3*b4
# state3 is a state after these two BSs
state3 = state2
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')

state3 = state3.subs(a1, (t2*b1 + 1j*r2*b2))
state3 = state3.subs(a2, (t3*b3 + 1j*r3*b4))

state3 = sp.expand(state3)

print('State 3:', state3)

# Consider ideal detectors first
# Both detector were clicked - Det1 and Det2.

# state4 is a state after measurement
state_4pre = 0

state4 = measure_state(state3, clicked=DET_CONF)

print('State 4:', state4)

# Now mixing state in a fourth BS
# Final state is state5
# b2 -> t4*a1 + 1j*t4*a2
# b4 -> t4*a2 + 1j*t4*a1
state5 = state4

state5 = state5.subs(b2, (t4*a1 + 1j*r4*a2))
state5 = state5.subs(b4, (t4*a2 + 1j*r4*a1))

state5 = sp.expand(state5)

print('State 5:', state5)

# Plotting final state.
# Matrix of coefficients.
state5_coeffs = get_state_coeffs(state5, max_power)

plot_state(state5_coeffs, 'Final State',  size=8, value='abs')
plot_state(state5_coeffs, 'Final State',  size=8, value='real')
plot_state(state5_coeffs, 'Final State',  size=8, value='imag')

'''
# plot input states
plt.bar(list(range(len(input_st))), input_st, width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], width=1, edgecolor='c')
plt.title('Input state')
plt.xlabel('Number of photons')
plt.show()
plt.bar(list(range(len(auxiliary_st))), auxiliary_st, color='g', width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], color='g', width=1, edgecolor='c')
plt.title('Auxiliary state')
plt.xlabel('Number of photons')
plt.show()
'''

# save setup configuration in file TODO
