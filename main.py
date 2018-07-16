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
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 4
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!a

# INPUT
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

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


#############
# Several params:
print('Started:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# First and last BS grids
r1_grid = 1
r4_grid = 11

bs1_even = True

# Phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

log_entropy_subs1_array = np.zeros((r4_grid, r1_grid), dtype=complex)
log_entropy_subs2_array = np.zeros((r4_grid, r1_grid), dtype=complex)
lin_entropy_subs1 = np.zeros((r4_grid, r1_grid), dtype=complex)
lin_entropy_subs2 = np.zeros((r4_grid, r1_grid), dtype=complex)
log_negativity = np.zeros((r4_grid, r1_grid), dtype=complex)
mut_information = np.zeros((r4_grid, r1_grid), dtype=complex)
full_fn_entropy = np.zeros((r4_grid, r1_grid), dtype=complex)
sqeez_dX1 = np.zeros((r4_grid, r1_grid), dtype=complex)
sqeez_dX2 = np.zeros((r4_grid, r1_grid), dtype=complex)

# log_negativity_aftdet = np.zeros((r4_grid, r1_grid), dtype=complex)

# Varying last BS (BS4)
T4_array = np.linspace(0, 1, r4_grid)
t4_array = np.sqrt(T4_array)

r4_fun = lambda tt: sqrt(1 - pow(tt, 2))
r4_vect_func = np.vectorize(r4_fun)
r4_array = r4_vect_func(t4_array)

# Varying first BS (BS1)
T1_array = np.linspace(0, 1, r1_grid)
t1_array = np.sqrt(T1_array)

r1_fun = lambda tt: sqrt(1 - pow(tt, 2))
r1_vect_func = np.vectorize(r1_fun)
r1_array = r1_vect_func(t1_array)

# Set BS1 - 50:50
if r1_grid is 1 and bs1_even:
    t1_array = [sqrt(0.5)]
    r1_array = [sqrt(1 - pow(t1_array[0], 2))]


# transmit all near detection area
# t2 = 0.99999
# r2 = sqrt(1 - t2**2)
# t3 = 0.99999
# r3 = sqrt(1 - t3**2)


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
        # Gives not normalised state
        state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
        # Calculating the norm
        norm_after_det = state_norm(state_after_dett_unappl)
        print('Norm after det.:', norm_after_det)
        # normalised state
        state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det

        # Build dens matrix and trace
        dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm)

        # The new method, works
        # dens_matrix_2channels = dens_matrix_with_trace_new(state_after_dett_unappl_norm, state_after_dett_unappl_norm)

        # Phase modulation
        dens_matrix_2channels_withph = phase_modulation(dens_matrix_2channels, phase_diff)

        # Disable a phase addition.
        # dens_matrix_2channels_withph = dens_matrix_2channels

        # after detection, with phase
        # log_negativity_aftdet[i] = negativity(dens_matrix_2channels_withph, neg_type='logarithmic')

        # Traced matrix after detection
        # afterdet_traced = trace_channel(dens_matrix_2channels, channel=4)

        # Transformation at last BS
        # Trim for better performance,
        # trim_size=10 for series_len=10
        # trim_size=4 for series_len=3
        trim_size = 5
        final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels_withph[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

        # Trace one channel out of final state
        final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
        print('trace of final reduced matrix 2nd channel:', np.trace(final_traced_subs1))

        # Other channel traced
        final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)
        print('trace of final reduced matrix 4th channel:', np.trace(final_traced_subs2))

        # Calculate entropy
        log_entanglement_subs1 = log_entropy(final_traced_subs1)
        log_entanglement_subs2 = log_entropy(final_traced_subs2)
        # print('FN entropy subs1: ', np.real(log_entanglement_subs1))
        # print('FN entropy subs2: ', np.real(log_entanglement_subs2))
        log_entropy_subs1_array[i, j] = log_entanglement_subs1
        log_entropy_subs2_array[i, j] = log_entanglement_subs2

        # Logarithmic entropy difference
        # print('FN entropy difference: ', log_entanglement_subs1 - log_entanglement_subs2)

        # Full entropy and the mutual information
        final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
        full_entr = log_entropy(final_reorg_matr)

        mut_information[i, j] = log_entanglement_subs1 + log_entanglement_subs2 - full_entr
        full_fn_entropy[i, j] = full_entr

        # lin_entropy_subs1[i, j] = np.real(linear_entropy(log_entanglement_subs1))
        # lin_entropy_subs2[i, j] = np.real(linear_entropy(log_entanglement_subs2))
        # print('Lin. entropy subs1: ', lin_entropy_subs1[i, j])
        # print('Lin. entropy subs2: ', lin_entropy_subs2[i, j])

        # Linear entropy difference
        print('Linear entropy difference: ', lin_entropy_subs1[i, j] - lin_entropy_subs2[i, j])

        log_negativity[i, j] = negativity(final_dens_matrix, neg_type='logarithmic')
        print('Log. negativity: ', log_negativity[i, j])

        dX1, dX2 = squeezing_quadratures(final_dens_matrix, channel=1)
        print('dX1:', dX1, ' dX2:', dX2)
        sqeez_dX1[i, j] = dX1
        sqeez_dX2[i, j] = dX2


# Varying t4
plt.plot(np.square(t4_array), np.real(log_entropy_subs1_array[:, 0]), label=r'$Log. FN \ entropy \ subs \ 1$')
plt.plot(np.square(t4_array), np.real(log_entropy_subs2_array[:, 0]), label=r'$Log. FN \ entropy \ subs \ 2$')
plt.plot(np.square(t4_array), np.real(mut_information[:, 0]), label=r'$Mut \ information, \ FN \ log. \ entropy.$')
plt.plot(np.square(t4_array), np.real(full_fn_entropy[:, 0]), label=r'$Full \ FN \ log. \ entropy.$')
# plt.plot(np.square(t4_array), np.real(lin_entropy[:, 0]), label=r'$Lin. entropy$')
plt.plot(np.square(t4_array), np.real(log_negativity[:, 0]), label=r'$Log. \ negativity$')
# plt.plot(np.square(t4_array), np.real(log_negativity_aftdet[:, 0]), label=r'$Log. negativity, after det. with phase$')
# plt.title(f'Entanglement. Phase=%0.2f pi. Det. - %s' % (ph_inpi, DET_CONF))
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel('$Entanglement$')
# plt.title('Phase = {0}pi'.format(ph_inpi))
plt.xlim([0, 1])
plt.legend()
plt.grid(True)
plt.show()


# plot squeezing
plt.plot(np.square(t4_array), sqeez_dX1[:, 0], label=r'$\Delta X_1$')
plt.plot(np.square(t4_array), sqeez_dX2[:, 0], label=r'$\Delta X_2$')
plt.plot(np.square(t4_array), np.multiply(sqeez_dX1[:, 0], sqeez_dX2[:, 0]), label=r'$\Delta X_1 \Delta X_2$')
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel('$Squeezing$')
plt.title('$Squeezing$')
plt.xlim([0, 1])
plt.legend()
plt.grid(True)
plt.show()
