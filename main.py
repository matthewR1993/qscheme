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
from time import gmtime, strftime

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
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
# auxiliary_st = single_photon(series_length)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
# DET_CONF = 'FIRST'  # 1st detector clicked
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

#
# # First BS
# state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)
#
# # Second and third BSs
# state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)
#
# # Applying detection operator. Receiving unnormalized state.
# state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
#
# # Calculate norm
# norm_after_det = state_norm(state_after_dett_unappl)
#
# # Apply normalisation
# state_after_dett_unappl_norm = state_after_dett_unappl/norm_after_det
#
# # Form dens matrix and trace. Dens. matrix for two channels
# dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm)
#
# channel2_densmatrix = trace_channel(dens_matrix_2channels, channel=4)
#
# # 2D  bars picture
# # plt.matshow(np.abs(channel2_densmatrix[:7, :7]))
# # plt.colorbar()
# # plt.title(r'$|\rho_{m n}| - after \ detection$')
# # plt.xlabel('m')
# # plt.ylabel('n')
# # plt.show()
#
# # 3D bars picture
# # data_array = np.array(np.abs(channel2_densmatrix[:7, :7]))
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]), np.arange(data_array.shape[0]))
# # x_data = x_data.flatten()
# # y_data = y_data.flatten()
# # z_data = data_array.flatten()
# # ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='#00ceaa', shade=True)
# # plt.title(r'$|\rho_{m n}| - after \ detection$')
# # plt.xlabel('m')
# # plt.ylabel('n')
# # plt.show()
#
#
# trim_size = 8
# final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)
#
# final_traced = trace_channel(final_dens_matrix, channel=4)
#
# final_traced_4 = trace_channel(final_dens_matrix, channel=2)
#
# # 2D bars picture
# # plt.matshow(np.abs(final_traced[:7, :7]))
# # plt.colorbar()
# # plt.title(r'$|\rho_{m n}| - output$')
# # plt.xlabel('m')
# # plt.ylabel('n')
# # plt.show()
#
# # 3D bars picture
# # data_array = np.array(np.abs(final_traced[:7, :7]))
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]), np.arange(data_array.shape[0]))
# # x_data = x_data.flatten()
# # y_data = y_data.flatten()
# # z_data = data_array.flatten()
# # ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='#00ceaa', shade=True)
# # plt.title(r'$|\rho_{m n}| - output$')
# # plt.xlabel('m')
# # plt.ylabel('n')
# # plt.show()
#
#
# print('Log. entropy: ', log_entropy(final_traced))
#
# print('Lin. entropy: ', np.real(linear_entropy(final_traced)))
#
# print('Log. negativity: ', negativity(final_dens_matrix, neg_type='logarithmic'))
#



################################
# Several params:
print('Started:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

r1_grid = 16
r4_grid = 16

bs1_even = True

log_entropy_array = np.zeros((r4_grid, r1_grid), dtype=complex)
lin_entropy = np.zeros((r4_grid, r1_grid), dtype=complex)
log_negativity = np.zeros((r4_grid, r1_grid), dtype=complex)

# Varying last BS (BS4)
a4 = 0
t4_array = np.linspace(0, 1, r4_grid)

r4_fun = lambda tt: sqrt(1 - pow(tt, 2) - pow(a4, 2))
r4_vect_func = np.vectorize(r4_fun)
r4_array = r4_vect_func(t4_array)

# Varying first BS (BS1)
a1 = 0
t1_array = np.linspace(0, 1, r1_grid)

r1_fun = lambda tt: sqrt(1 - pow(tt, 2) - pow(a1, 2))
r1_vect_func = np.vectorize(r1_fun)
r1_array = r1_vect_func(t1_array)

# Set BS1 - 50:50
if r1_grid is 1 and bs1_even:
    t1_array = [sqrt(0.5)]
    r1_array = [sqrt(1 - pow(t1_array[0], 2) - pow(a1, 2))]

# loops
for i in range(r4_grid):
    t4 = t4_array[i]
    r4 = r4_array[i]
    for j in range(r1_grid):
        print('step t4:', i)
        print('step t1:', j)

        t1 = t1_array[j]
        r1 = r1_array[j]

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

        # Traced matrix after detection
        # afterdet_traced = trace_channel(dens_matrix_2channels, channel=4)

        # Transformation at last BS
        trim_size = 8
        final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

        # Trace one channel out of final state
        final_traced = trace_channel(final_dens_matrix, channel=4)
        print('Trace of final matrix:', np.trace(final_traced))

        # Calculate entropy
        log_entanglement = log_entropy(final_traced)
        # log_entanglement = log_entropy(afterdet_traced)
        print('Log. entropy: ', np.real(log_entanglement))
        log_entropy_array[i, j] = log_entanglement

        lin_entropy[i, j] = np.real(linear_entropy(final_traced))
        print('Lin. entropy: ', lin_entropy[i, j])

        log_negativity[i, j] = negativity(final_dens_matrix, neg_type='logarithmic')
        print('Log. negativity: ', log_negativity[i, j])


# Varying t4
plt.plot(t4_array, np.real(log_entropy_array[:, 0]), label=r'$Log. entropy$')
plt.plot(t4_array, np.real(lin_entropy[:, 0]), label=r'$Lin. entropy$')
plt.plot(t4_array, np.real(log_negativity[:, 0]), label=r'$Log. negativity$')
plt.title(r'Entanglement.')
plt.xlabel(r'$t_{4}$', fontsize=16)
plt.xlim([0, 1])
plt.legend()
plt.grid(True)
plt.show()


# mult = 1e-3
#
# plt.plot(t4_array, np.real(log_entropy_array), label=r'$Log. entropy$')
# plt.plot(t4_array, np.real(lin_entropy), label=r'$Lin. entropy$')
# plt.plot(t4_array, np.real(mult*log_negativity), label=r'$Log. negativity*10^{-3}$')
# plt.title(r'Entanglement.')
# plt.xlabel(r'$t_{4}$', fontsize=16)
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()


# TODO S(t1, t4) plot
fig = plt.figure()
ax = fig.gca(projection='3d')

X = t1_array
Y = t4_array
X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, np.real(log_entropy_array), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


'''

# plot input states
plt.bar(list(range(len(input_st))), input_st, width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], width=1, edgecolor='c')
plt.title('Input state')
plt.xlabel('Number of photons')
plt.show()
plt.bar(list(range(len(auxiliar y_st))), auxiliary_st, color='g', width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], color='g', width=1, edgecolor='c')
plt.title('Auxiliary state')
plt.xlabel('Number of photons')
plt.show()

'''
