# Checking influence of absorption as position of channels with loses.

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
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum, squeezed_coherent_state
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 3
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT
input_st = single_photon(input_series_length)
# input_st = coherent_state(input_series_length, alpha=1)
# input_st = squeezed_vacuum(input_series_length, squeezing_amp=0.5, squeezing_phase=0)
# input_st = squeezed_coherent_state(input_series_length, alpha=1, squeezing_amp=0.5, squeezing_phase=0)
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

# First, channels with loses are located before BS

t1 = sqrt(0.5)
r1 = sqrt(1 - pow(t1, 2))

t2 = sqrt(0.5)
r2 = sqrt(1 - pow(t2, 2))

t3 = sqrt(0.5)
r3 = sqrt(1 - pow(t3, 2))

state_aft2bs_unappl = two_bs2x4_transform(t1, r1, t2, r2, mut_state_unappl)

# Make state appl
state_aft2bs_appl = make_state_appliable_4ch(state_aft2bs_unappl)

# Form density matrix and trace
dm = dens_matrix_4ch(state_aft2bs_appl)

# Trace loosy channels
size = len(dm)
dm_aft_trace_appl = np.zeros((size,) * 4, dtype=complex)

for p1 in range(size):
    for p1_ in range(size):
        for p2 in range(size):
            for p2_ in range(size):
                matrix_sum = 0
                for k3 in range(size):
                    for k4 in range(size):
                        matrix_sum = matrix_sum + dm[p1, p2, k3, k4, p1_, p2_, k3, k4]
                dm_aft_trace_appl[p1, p2, p1_, p2_] = matrix_sum


# last BS transformation
final_dens_matrix = bs_densmatrix_transform(dm_aft_trace_appl, t3, r3)


###############
# Second method
# First, channels with loses are located after BS
state_aft_1st_bs_unappl = bs2x2_transform(t3, r3, mut_state_unappl)

state_aft2bs_unappl_2 = two_bs2x4_transform(t1, r1, t2, r2, state_aft_1st_bs_unappl)

# Make state appl
state_aft2bs_appl_2 = make_state_appliable_4ch(state_aft2bs_unappl_2)

# Form density matrix and trace
dm_2 = dens_matrix_4ch(state_aft2bs_appl_2)

# Trace loosy channels
size = len(dm_2)
dm_aft_trace_appl_2 = np.zeros((size,) * 4, dtype=complex)

for p1 in range(size):
    for p1_ in range(size):
        for p2 in range(size):
            for p2_ in range(size):
                matrix_sum = 0
                for k3 in range(size):
                    for k4 in range(size):
                        matrix_sum = matrix_sum + dm_2[p1, p2, k3, k4, p1_, p2_, k3, k4]
                dm_aft_trace_appl_2[p1, p2, p1_, p2_] = matrix_sum


matr_diff = dm_aft_trace_appl_2 - final_dens_matrix[:5, :5, :5, :5]

# prob distr diff
pd1 = prob_distr(final_dens_matrix[:5, :5, :5, :5])

pd2 = prob_distr(dm_aft_trace_appl_2)


