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
series_length = 3
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!a

# INPUT
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector clicked
# DET_CONF = 'THIRD'  # 3rd detector clicked
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
r4_grid = 19

bs1_even = True

# Phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

log_entropy_array = np.zeros((r4_grid, r1_grid), dtype=complex)
lin_entropy = np.zeros((r4_grid, r1_grid), dtype=complex)
log_negativity = np.zeros((r4_grid, r1_grid), dtype=complex)

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
        trim_size = 4
        final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels_withph[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

        # Trace one channel out of final state
        final_traced = trace_channel(final_dens_matrix, channel=4)
        print('trace of final reduced matrix 2nd channel:', np.trace(final_traced))

        # Other channel traced
        final_traced_4th = trace_channel(final_dens_matrix, channel=2)
        print('trace of final reduced matrix 4th channel:', np.trace(final_traced_4th))

        # Calculate entropy
        # log_entanglement = log_entropy(final_traced)
        log_entanglement = log_entropy(final_traced_4th)  # other channel traced matrix
        print('FN entropy: ', np.real(log_entanglement))
        log_entropy_array[i, j] = log_entanglement

        # Logarithmic entropy difference
        print('FN entropy difference: ', log_entanglement - log_entropy(final_traced_4th))

        # lin_entropy[i, j] = np.real(linear_entropy(final_traced))
        lin_entropy[i, j] = np.real(linear_entropy(final_traced_4th))  # other channel traced matrix
        print('Lin. entropy: ', lin_entropy[i, j])

        # Linear entropy difference
        print('Linear entropy difference: ', lin_entropy[i, j] - linear_entropy(final_traced_4th))

        log_negativity[i, j] = negativity(final_dens_matrix, neg_type='logarithmic')
        print('Log. negativity: ', log_negativity[i, j])


# Varying t4
plt.plot(np.square(t4_array), np.real(log_entropy_array[:, 0]), label=r'$Log. FN \ entropy$')
# plt.plot(np.square(t4_array), np.real(lin_entropy[:, 0]), label=r'$Lin. entropy$')
plt.plot(np.square(t4_array), np.real(log_negativity[:, 0]), label=r'$Log. negativity$')
# plt.plot(np.square(t4_array), np.real(log_negativity_aftdet[:, 0]), label=r'$Log. negativity, after det. with phase$')
plt.title(f'Entanglement. Phase=%0.2f pi. Det. - %s' % (ph_inpi, DET_CONF))
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel('$Entanglement$')
# plt.title('Phase = {0}pi'.format(ph_inpi))
plt.xlim([0, 1])
plt.legend()
plt.grid(True)
plt.show()
