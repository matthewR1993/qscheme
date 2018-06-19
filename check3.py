# A coherent state and a single photon with two beam splitters and phase modul.
# Gives zeros entanglement for two coherent states

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
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 14
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT
# input_st = single_photon(input_series_length)
# input_st = coherent_state(input_series_length, alpha=1)
input_st = squeezed_vacuum(input_series_length, squeezing_amp=0.5, squeezing_phase=0)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(auxiliary_series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))


in_state_tf = tf.constant(input_st, tf.complex128)
aux_state_tf = tf.constant(auxiliary_st, tf.complex128)

# A tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)


# The first BS
state_after_bs1_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

grd = 10

# Varying BS2
t2_arr = np.linspace(0, 1, grd)
r2_arr = np.zeros(grd)
for i in range(grd):
    r2_arr[i] = sqrt(1 - pow(t2_arr[i], 2))

ph_inpi = 0.5
phase_mod = ph_inpi * np.pi

log_entr_arr = np.zeros(grd)
log_neg_arr = np.zeros(grd)

for i in range(grd):
    print('step:', i)
    t2 = t2_arr[i]
    r2 = r2_arr[i]
    # The phase modulation
    state_after_phmod_unappl = phase_modulation_state(state_after_bs1_unappl, phase_mod)

    # The second BS
    state_after_bs2_unappl = bs2x2_transform(t2, r2, state_after_phmod_unappl)

    state_after_bs2_appl = make_state_appliable(state_after_bs2_unappl)

    dens_matrix_2channels = dens_matrix(state_after_bs2_appl)

    reduced_dens_matrix = trace_channel(dens_matrix_2channels, channel=2)

    # Entanglement
    log_fn_entropy = log_entropy(reduced_dens_matrix)
    log_entr_arr[i] = log_fn_entropy
    # print('FN log. entropy:', log_fn_entropy)

    log_negativity = negativity(dens_matrix_2channels, neg_type='logarithmic')
    log_neg_arr[i] = log_negativity
    # print('Log. negativity', log_negativity)


fig, ax = plt.subplots()
ax.plot(np.square(t2_arr), log_entr_arr, label=r'$Log. FN \ entropy$')
ax.plot(np.square(t2_arr), log_neg_arr, label=r'$Log. negativity$')
plt.title('Phase = {0}pi'.format(ph_inpi))
plt.xlabel('$T_{2}$')
plt.ylabel('$Entanglement$')
plt.legend()
plt.grid(True)
plt.show()


# Varying phase
phase_arr = np.linspace(0, np.pi, grd)

log_entr_arr2 = np.zeros(grd)
log_neg_arr2 = np.zeros(grd)

t2 = sqrt(0.999)
r2 = sqrt(1 - t2**2)

for i in range(grd):
    print('step:', i)
    phase_mod = phase_arr[i]

    # The phase modulation
    state_after_phmod_unappl = phase_modulation_state(state_after_bs1_unappl, phase_mod)

    # The second BS
    state_after_bs2_unappl = bs2x2_transform(t2, r2, state_after_phmod_unappl)

    state_after_bs2_appl = make_state_appliable(state_after_bs2_unappl)

    dens_matrix_2channels = dens_matrix(state_after_bs2_appl)

    reduced_dens_matrix = trace_channel(dens_matrix_2channels, channel=2)

    # Entanglement
    log_fn_entropy = log_entropy(reduced_dens_matrix)
    log_entr_arr2[i] = log_fn_entropy
    # print('FN log. entropy:', log_fn_entropy)

    log_negativity = negativity(dens_matrix_2channels, neg_type='logarithmic')
    log_neg_arr2[i] = log_negativity
    # print('Log. negativity', log_negativity)


fig, ax = plt.subplots()
ax.plot(phase_arr, log_entr_arr2, label=r'$Log. FN \ entropy$')
ax.plot(phase_arr, log_neg_arr2, label=r'$Log. negativity$')
plt.xlabel('$phase$')
plt.ylabel('$Entanglement$')
ax.set_xticks([0, 0.5*np.pi, np.pi])
ax.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
plt.legend()
plt.grid(True)
plt.show()
