import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from qutip import (wigner, super_tensor, Qobj)

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *


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

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector clicked
# DET_CONF = 'THIRD'  # 3rd detector clicked
# DET_CONF = 'NONE'  # None of detectors was clicked

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

# diag_factorials = diagonal_factorials(input_series_length)
# diag_factorials_tf = tf.constant(diag_factorials, tf.float64)
# in_state_tf_appl = tf.einsum('mn,n->n', diag_factorials_tf, in_state_tf)
# aux_state_tf_appl = tf.einsum('mn,n->n', diagl_factorials_tf, aux_state_tf)

# tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)


# First BS
state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

# Second and third BSs
state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)

# Applying detection operator. Receiving unnormalised state.
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)

# Calculate norm
norm_after_det = state_norm(state_after_dett_unappl)

# Apply normalisation
state_after_dett_unappl_norm = state_after_dett_unappl/norm_after_det

# Form dens matrix and trace. Dens. matrix for two channels
dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm)

channel2_densmatrix = trace_channel(dens_matrix_2channels, channel=4)

# 2D  bars picture
# plt.matshow(np.abs(channel2_densmatrix[:7, :7]))
# plt.colorbar()
# plt.title(r'$|\rho_{m n}| - after \ detection$')
# plt.xlabel('m')
# plt.ylabel('n')
# plt.show()

# 3D bars picture
data_array = np.array(np.abs(channel2_densmatrix[:7, :7]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]), np.arange(data_array.shape[0]))
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = data_array.flatten()
ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='#00ceaa', shade=True)
plt.title(r'$|\rho_{m n}| - after \ detection$')
plt.xlabel('m')
plt.ylabel('n')
plt.show()


trim_size = 8
final_dens_matrix = last_bs(dens_matrix_2channels[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

final_traced = trace_channel(final_dens_matrix, channel=4)

final_traced_4 = trace_channel(final_dens_matrix, channel=2)

# TODO trace 2d channel and compare matrix with traced from 4th

# 2D bars picture
# plt.matshow(np.abs(final_traced[:7, :7]))
# plt.colorbar()
# plt.title(r'$|\rho_{m n}| - output$')
# plt.xlabel('m')
# plt.ylabel('n')
# plt.show()

# 3D bars picture
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


log_entanglement = log_entropy(final_traced)
print('Log. entropy: ', np.real(log_entanglement))

afterdet_traced = trace_channel(dens_matrix_2channels, channel=4)

# TODO check trace after last BS
# np.trace(final_traced)
# np.trace(afterdet_traced)


# afterdet_traced
# final_traced

afterdet_qtip = Qobj(afterdet_traced)

afterdet_sqrt = afterdet_qtip.sqrtm()

afterdet_sqrt_np = afterdet_sqrt.full()

assembled_afterdet = afterdet_sqrt_np @ afterdet_sqrt_np

mat_diff = afterdet_traced - assembled_afterdet

mat_diff_norm = np.abs(mat_diff).sum()

# for 4 tensor, not traced
afterdet4_qtip = super_tensor(dens_matrix_2channels)




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


# Wigner function
# xvec = np.linspace(-5, 5, 200)
# wig_fun = wigner(Qobj(final_traced), xvec, xvec)

# plt.plot_surface(xvec, xvec, wig_fun)
# plt.colorbar()
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(xvec, xvec)
# surf = ax.plot_surface(X, Y, wig_fun, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


# Several params:
grid = 20

log_entropy_array = np.zeros(grid, dtype=complex)
lin_entropy = np.zeros(grid, dtype=complex)
log_negativity = np.zeros(grid, dtype=complex)

# Varying last BS
a4 = 0
t4_array = np.linspace(0, 1, grid)

r4_fun = lambda tt: sqrt(1 - pow(tt, 2) - pow(a4, 2))
r4_vect_func = np.vectorize(r4_fun)
r4_array = r4_vect_func(t4_array)

# loops
for i in range(grid):
    print('step:', i)
    t4 = t4_array[i]
    r4 = r4_array[i]

    # First BS
    state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

    # 2d and 3rd BS
    state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)

    # Detection
    # unnormalised state
    state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
    norm_after_det = state_norm(state_after_dett_unappl)
    # normalised state
    state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det

    # Dens matrix and trace
    dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm)

    # traced matrix after detection
    # afterdet_traced = trace_channel(dens_matrix_2channels, channel=4)

    # Transformation at last BS
    trim_size = 8
    final_dens_matrix = last_bs(dens_matrix_2channels[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

    # Trace one channel out of final state
    final_traced = trace_channel(final_dens_matrix, channel=4)

    # Calculate entropy
    log_entanglement = log_entropy(final_traced)
    # log_entanglement = log_entropy(afterdet_traced)
    print('Log. entropy: ', np.real(log_entanglement))
    log_entropy_array[i] = log_entanglement

    lin_entropy[i] = linear_entropy(final_traced)


plt.plot(t4_array, np.real(log_entropy_array))
plt.title(r'$S = -\sum_{n} \lambda_{n} ln(\lambda_{n})$')
plt.xlabel(r'$t_{4}$')
plt.ylabel('S')
plt.xlim([0, 1])
plt.show()

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
