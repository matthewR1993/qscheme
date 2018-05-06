import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf


sess = tf.Session()

# Parameters for states
series_length = 9
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!

# INPUT
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
# auxiliary_st = single_photon(series_length)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement detectors configuration
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector clicked
# DET_CONF = 'THIRD'  # 3rd detector clicked
# DET_CONF = 'NONE'  # None of detectors was clicked

# diag_factorials = diagonal_factorials(input_series_length)

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

# diag_factorials_tf = tf.constant(diag_factorials, tf.float64)
# in_state_tf_appl = tf.einsum('mn,n->n', diag_factorials_tf, in_state_tf)
# aux_state_tf_appl = tf.einsum('mn,n->n', diagl_factorials_tf, aux_state_tf)

# tensor product, return numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)


# better
state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)

# unnormalised
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event='FIRST')

# norm_before_det = state_norm(state_aft2bs_unappl)
norm_after_det = state_norm(state_after_dett_unappl)

# normalised
state_after_dett_unappl_norm = state_after_dett_unappl/norm_after_det

dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm )

channel2_densmatrix = trace_channel(dens_matrix_2channels, channel=4)

# plt.matshow(np.abs(channel2_densmatrix[:7, :7]))
# plt.colorbar()
# plt.title(r'$|\rho_{m n}| - after \ detection$')
# plt.xlabel('m')
# plt.ylabel('n')
# plt.show()
#
# 3d picture
# data_array = np.array(np.abs(channel2_densmatrix[:7, :7]))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]), np.arange(data_array.shape[0]))
# x_data = x_data.flatten()
# y_data = y_data.flatten()
# z_data = data_array.flatten()
# ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='#00ceaa', shade=True)
# plt.title(r'$|\rho_{m n}| - after \ detection$')
# plt.xlabel('m')
# plt.ylabel('n')
# plt.show()


trim_size = 8
final_dens_matrix = last_bs(dens_matrix_2channels[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

final_traced = trace_channel(final_dens_matrix, channel=4)


# plots
plt.matshow(np.abs(final_traced[:7, :7]))
plt.colorbar()
plt.title(r'$|\rho_{m n}| - output$')
plt.xlabel('m')
plt.ylabel('n')
plt.show()

# 3d picture
data_array = np.array(np.abs(final_traced[:7, :7]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]), np.arange(data_array.shape[0]))
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = data_array.flatten()
ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='#00ceaa', shade=True)
plt.title(r'$|\rho_{m n}| - output$')
plt.xlabel('m')
plt.ylabel('n')
plt.show()


prob_distr_matrix = prob_distr(final_dens_matrix)

plt.matshow(np.real(prob_distr_matrix[:6, :6]))
plt.colorbar()
plt.title(r'$P_{m n}$')
plt.xlabel('m')
plt.ylabel('n')
plt.show()


log_entanglement = log_entropy(final_traced)
print('Log. entropy: ', np.real(log_entanglement))


# TODO
# # Entanglement negativity, input - matrix of 2 channels
# def calculate_negativity(dens_matrix):
#     neg = 0
#     part_transposed = partial_transpose(dens_matrix)
#     w, v = np.linalg.eig(np.diag((1, 2, 3)))
#     return neg
#
#
# part_transposed = partial_transpose(final_dens_matrix)
# w, v = np.linalg.eig(part_transposed)
# w  # values


# TODO calculate Wigner function



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

# old method

# Setting up state before first BS.
# a1, a2 = sp.symbols('a1 a2')
# g = 0
# for i in range(len(input_st)):
#     g = g + input_st[i]*(a1**i)
# f = 0
# for i in range(len(auxiliary_st)):
#     f = f + auxiliary_st[i]*(a2**i)

# g(a1) - input
# f(a2) - auxiliary
# Initial state = g(a1) * f(a2)
# state1 = g * f

#state1_coeffs_unapp = get_state_coeffs(sp.expand(state1), max_power + 1, operators_form='unapplied')

#plot_state(state1_coeffs_unapp, 'Initial State',  size=10, value='real')

# State after mixing at first BS
# state2 = state1
# b1, b2 = sp.symbols('b1 b2')

# a1 -> t1*a1 + 1j*r1*a2
# state2 = state2.subs(a1, (t1*b1 + 1j*r1*b2))
# state2 = state2.subs(a2, (t1*b2 + 1j*r1*b1))

# put 'a' operators back
# state2 = state2.subs(b1, a1)
# state2 = state2.subs(b2, a2)

# state2 = sp.expand(state2)
# print('State 2:', state2)

# Plot state2
# state2_coeffs = get_state_coeffs(state2, max_power + 1, operators_form='unapplied')

# plot_state(state2_coeffs, 'State2',  size=8, value='abs')
# plot_state(state2_coeffs, 'State2',  size=8, value='real')
# plot_state(state2_coeffs, 'State2',  size=8, value='imag')

# 'state2' is a state after BS

# a1 goes to 2nd BS with t2, r2 and split into b1 and b2. Therefore: a1 -> t2*b1 + 1j*r2*b2
# a2 goes to 3rd BS with t3, r3 and split into b3 and b4. Therefore: a2 -> t3*b3 + 1j*r3*b4
# state3 is a state after these two BSs
# state3 = state2
# b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')

# state3 = state3.subs(a1, (t2*b2 + 1j*r2*b1))
# state3 = state3.subs(a2, (t3*b4 + 1j*r3*b3))

# state3 = sp.expand(state3)

# print('State 3:', state3)


# state4 is a state after measurement
# state_4pre = 0
# state4 = measure_state(state3, clicked=DET_CONF)
# print('State 4:', state4)



# Now mixing state in a fourth BS
# Final state is state5
# b2 -> t4*a1 + 1j*t4*a2
# b4 -> t4*a2 + 1j*t4*a1
#state5 = state4

#state5 = state5.subs(b2, (t4*a1 + 1j*r4*a2))
#state5 = state5.subs(b4, (t4*a2 + 1j*r4*a1))

#state5 = sp.expand(state5)

#print('State 5:', state5)

# Plotting final state.
# Matrix of coefficients.
#state5_coeffs = get_state_coeffs(state5, max_power)

#plot_state(state5_coeffs, 'Final State',  size=8, value='abs')
#plot_state(state5_coeffs, 'Final State',  size=8, value='real')
#plot_state(state5_coeffs, 'Final State',  size=8, value='imag')
