# Measuring functions performance

import sys
import time
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import tensorflow as tf
import scipy
from time import gmtime, strftime

from customutils.utils import *
from core.basic import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel
# auxiliary_st = single_photon(series_length)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

# Building a mutual state via tensor product, that returns numpy array.
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# Start time.
print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# The first and the last BS grids.
r1_grid = 1
r4_grid = 11

# If BS1 is 50:50 symetrical.
bs1_is_symmetrical = True

# The phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

# BS2, BS3.
t2 = sqrt(0.9)
r2 = sqrt(1 - t2**2)
t3 = sqrt(0.1)
r3 = sqrt(1 - t3**2)

# Measurements start here.

# First BS.
start = time.time()
state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)
end = time.time()
print('First BS time:', end - start)


# 2d and 3rd BS.
start = time.time()
state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)
end = time.time()
print('BS 2 and 3 time:', end - start)


# The detection event.
start = time.time()
# Gives non-normalised state.
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
# Calculating the norm.
norm_after_det = state_norm(state_after_dett_unappl)
print('Norm after det.:', norm_after_det)
# The normalised state.
state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det
end = time.time()
print('Detection:', end - start)


# Trim the state
start = time.time()
state_after_dett_unappl_norm_tr = trim_state(state_after_dett_unappl_norm, error=1e-9, start_index=12)
end = time.time()
print('Trim the state in 4 chann.:', end - start)


state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:10, :10, :10, :10]

# Old method, very slow! 110 sec.
# Building dens. matrix and trace.
start = time.time()
dens_matrix_2channels_old = dens_matrix_with_trace(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)
end = time.time()
print('Dens. metrix with trace, OLD:', end - start)

# A new method.
from core.optimized import transformations
start = time.time()
dens_matrix_2channels = transformations.dm_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm)
end = time.time()
print('Dens. metrix with trace, NEW:', end - start)

# Comparing an old and a new
np.testing.assert_allclose(dens_matrix_2channels_old, dens_matrix_2channels)


# Phase modulation
start = time.time()
dens_matrix_2channels_withph = phase_modulation(dens_matrix_2channels, phase_diff)
end = time.time()
print('Phase modulation:', end - start)


# TODO Slow! 69 sec.
trim_size = 10
start = time.time()
final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels_withph[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)
end = time.time()
print('BS4 density matrix transformation:', end - start)


start = time.time()
final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
end = time.time()
print('Trace one channel out of the final ds. matrix:', end - start)

start = time.time()
log_entanglement_subs1 = log_entropy(final_traced_subs1)
end = time.time()
print('Log. entropy:', end - start)

start = time.time()
final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
full_entr = log_entropy(final_reorg_matr)
end = time.time()
print('Full entropy:', end - start)

start = time.time()
negativity(final_dens_matrix, neg_type='logarithmic')
end = time.time()
print('Negativity:', end - start)

start = time.time()
# Squeezing quadratures.
dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
end = time.time()
print('Squezing quadratures:', end - start)

start = time.time()
# ERP correlations.
erp_x, erp_p = erp_squeezing_correlations(final_dens_matrix)
end = time.time()
print('EPR correlations:', end - start)
