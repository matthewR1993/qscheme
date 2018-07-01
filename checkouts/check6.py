# Similar to checkout3 but with detection

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
series_length = 10
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
# auxiliary_st = single_photon(series_length)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
# DET_CONF = 'FIRST'  # 1st detector clicked
# DET_CONF = 'THIRD'  # 3rd detector clicked
DET_CONF = 'NONE'  # None of detectors were clicked

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)


# tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# First BS
state_after_bs1_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

# Works
# get_state_norm_2ch(state_after_bs1_unappl)


# plt.matshow(np.abs(state_after_bs1_unappl))
# plt.xlabel('n')
# plt.ylabel('m')
# plt.colorbar()
# plt.show()

# 2nd and 3rd detectors
state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs1_unappl)


dens_matrix_2channels = dens_matrix_with_trace(state_aft2bs_unappl, state_aft2bs_unappl)

# Traced matrices
final_traced_2chan = trace_channel(dens_matrix_2channels, channel=4)
print('Trace of reduced matrix:', np.trace(final_traced_2chan))

# Other channel traced
final_traced_4chan = trace_channel(dens_matrix_2channels, channel=2)
print('trace of reduced matrix:', np.trace(final_traced_4chan))

# FN entropy for diff channels works, its the same
log_entanglement_4chan = log_entropy(final_traced_4chan)
log_entanglement_2chan = log_entropy(final_traced_2chan)

# Negativity
neg = negativity(dens_matrix_2channels, neg_type='logarithmic')

# TODO S(T1), LN(T1) at this point


# loop over T1
r1_grd = 11
T1_array = np.linspace(0, 1, r1_grd)
t1_array = np.sqrt(T1_array)

r1_fun = lambda tt: sqrt(1 - pow(tt, 2))
r1_vect_func = np.vectorize(r1_fun)
r1_array = r1_vect_func(t1_array)

neg_arr = np.zeros(r1_grd)
log_entropy_arr_2chan = np.zeros(r1_grd)
log_entropy_arr_4chan = np.zeros(r1_grd)

for i in range(r1_grd):
    print('step', i)
    t1 = t1_array[i]
    r1 = r1_array[i]
    # First BS
    state_after_bs1_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

    state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs1_unappl)

    dens_matrix_2channels = dens_matrix_with_trace(state_aft2bs_unappl, state_aft2bs_unappl)

    # Traced matrices
    final_traced_2chan = trace_channel(dens_matrix_2channels, channel=4)
    print('Trace of reduced matrix:', np.trace(final_traced_2chan))

    # Other channel traced
    final_traced_4chan = trace_channel(dens_matrix_2channels, channel=2)
    print('trace of reduced matrix:', np.trace(final_traced_4chan))

    # FN entropy for diff channels works, its the same
    log_entanglement_4chan = log_entropy(final_traced_4chan)
    log_entanglement_2chan = log_entropy(final_traced_2chan)

    print('diff channels entropy diff', log_entanglement_4chan - log_entanglement_2chan)

    # Negativity
    neg = negativity(dens_matrix_2channels, neg_type='logarithmic')

    neg_arr[i] = neg
    log_entropy_arr_4chan[i] = log_entanglement_4chan
    log_entropy_arr_2chan[i] = log_entanglement_2chan


fig, ax = plt.subplots()
ax.plot(np.square(t1_array), log_entropy_arr_2chan, label=r'$Log. FN \ entropy$')
ax.plot(np.square(t1_array), log_entropy_arr_4chan, label=r'$Log. FN \ entropy$')
ax.plot(np.square(t1_array), neg_arr, label=r'$Log. negativity$')
plt.title('Phase = {0}pi'.format(0))
plt.xlabel('$T_{2}$')
plt.ylabel('$Entanglement$')
plt.legend()
plt.grid(True)
plt.show()





# TODO calculate the whole scheme but without detection

# TODO Later this

# Gives not normalised state
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
# norm
norm_after_det = state_norm(state_after_dett_unappl)
# normalised state
state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det


