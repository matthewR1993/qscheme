# Sollution with a focused parameters grid.

import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/usr/local/lib/python3.5/dist-packages')

from time import gmtime, strftime
import tensorflow as tf
import numpy as np
import argparse

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state


sess = tf.Session()

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--det", help="Detection", type=str, required=True)
parser.add_argument("-p", "--phase", help="Phase in pi", type=float, required=True)
args = parser.parse_args()

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res15/'
# save_root = '/home/matthew/qscheme/results/res15/'
fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(args.phase, args.det)
print('Saving path:', save_root + fname)

# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel
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

# Building a mutual state via tensor product, that returns numpy array.
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

QUANT_T0_MINIMIZE = 'EPR_P'
SCALING_DEPTH = 2

# The phase difference before last BS
# ph_inpi = 0.0
ph_inpi = args.phase
phase_diff = ph_inpi * np.pi

# BS grids.
r1_grid = 11
r4_grid = 11

r2_grid = 11
r3_grid = 11


T1_min_arr = np.zeros(SCALING_DEPTH, dtype=complex)
T1_max_arr = np.zeros(SCALING_DEPTH, dtype=complex)
T4_min_arr = np.zeros(SCALING_DEPTH, dtype=complex)
T4_max_arr = np.zeros(SCALING_DEPTH, dtype=complex)

T2_min_arr = np.zeros(SCALING_DEPTH, dtype=complex)
T2_max_arr = np.zeros(SCALING_DEPTH, dtype=complex)
T3_min_arr = np.zeros(SCALING_DEPTH, dtype=complex)
T3_max_arr = np.zeros(SCALING_DEPTH, dtype=complex)

# Starting BS parameters grid range.
T1_min_arr[0] = 0.0
T1_max_arr[0] = 1.0
T4_min_arr[0] = 0.0
T4_max_arr[0] = 1.0

T2_min_arr[0] = 0.0001
T2_max_arr[0] = 0.9999
T3_min_arr[0] = 0.0001
T3_max_arr[0] = 0.9999


print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
# Reduction step iteration.
for p in range(SCALING_DEPTH):

    # BS values range.
    T1_min = T1_min_arr[p]
    T1_max = T1_max_arr[p]
    T4_min = T4_min_arr[p]
    T4_max = T4_max_arr[p]

    T2_min = T2_min_arr[p]
    T2_max = T2_max_arr[p]
    T3_min = T3_min_arr[p]
    T3_max = T3_max_arr[p]

    # Varying BSs.
    t1_array, r1_array = bs_params(T1_min, T1_max, r4_grid)
    t4_array, r4_array = bs_params(T4_min, T4_max, r4_grid)
    t2_array, r2_array = bs_params(T2_min, T2_max, r2_grid)
    t3_array, r3_array = bs_params(T3_min, T3_max, r3_grid)

    det_prob_array = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    log_entropy_subs1_array = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    log_entropy_subs2_array = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    lin_entropy_subs1 = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    lin_entropy_subs2 = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    log_negativity = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    mut_information = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    full_fn_entropy = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    sqeez_dX = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    sqeez_dP = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    epr_correl_x = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
    epr_correl_p = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)

    for n1 in range(r1_grid):
        for n4 in range(r4_grid):
            for n2 in range(r2_grid):
                for n3 in range(r3_grid):
                    print('Steps [n1, n4, n2, n3]:', n1, n4, n2, n3)
                    bs_params = {
                        't1': t1_array[n1],
                        'r1': r1_array[n1],
                        't4': t4_array[n4],
                        'r4': r4_array[n4],
                        't2': t2_array[n2],
                        'r2': r2_array[n2],
                        't3': t3_array[n3],
                        'r3': r3_array[n3],
                    }
                    final_dens_matrix, det_prob = process_all(mut_state_unappl, bs_params, phase_diff=phase_diff,
                                                              det_event=DET_CONF)

                    det_prob_array[n1, n4, n2, n3] = det_prob

                    # Trace one channel out of final state
                    final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)

                    # Other channel traced
                    final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)

                    # Calculate entropy
                    log_entanglement_subs1 = log_entropy(final_traced_subs1)
                    log_entanglement_subs2 = log_entropy(final_traced_subs2)
                    log_entropy_subs1_array[n1, n4, n2, n3] = log_entanglement_subs1
                    log_entropy_subs2_array[n1, n4, n2, n3] = log_entanglement_subs2

                    # Full entropy and the mutual information
                    final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
                    full_entr = log_entropy(final_reorg_matr)

                    mut_information[n1, n4, n2, n3] = log_entanglement_subs1 + log_entanglement_subs2 - full_entr
                    full_fn_entropy[n1, n4, n2, n3] = full_entr

                    log_negativity[n1, n4, n2, n3] = negativity(final_dens_matrix, neg_type='logarithmic')

                    # Squeezing quadratures.
                    dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
                    sqeez_dX[n1, n4, n2, n3] = dX
                    sqeez_dP[n1, n4, n2, n3] = dP

                    # ERP correlations.
                    epr_x, epr_p = erp_squeezing_correlations(final_dens_matrix)
                    epr_correl_x[n1, n4, n2, n3] = epr_x
                    epr_correl_p[n1, n4, n2, n3] = epr_p

    epr_x_min= np.amin(epr_correl_x)
    epr_p_min = np.amin(epr_correl_p)
    dX_min = np.amin(sqeez_dX)
    dP_min = np.amin(sqeez_dP)

    uncert = np.multiply(sqeez_dX, sqeez_dP)

    uncert_min_ind = list(np.unravel_index(np.argmin(uncert, axis=None), uncert.shape))
    dX_min_ind = list(np.unravel_index(np.argmin(sqeez_dX, axis=None), sqeez_dX.shape))
    dP_min_ind = list(np.unravel_index(np.argmin(sqeez_dP, axis=None), sqeez_dP.shape))
    epr_x_min_ind = list(np.unravel_index(np.argmin(epr_correl_x, axis=None), epr_correl_x.shape))
    epr_p_min_ind = list(np.unravel_index(np.argmin(epr_correl_p, axis=None), epr_correl_p.shape))

    # Calculate the minimun.
    if QUANT_T0_MINIMIZE is 'EPR_X':
        ind = epr_x_min_ind
    elif QUANT_T0_MINIMIZE is 'EPR_P':
        ind = epr_p_min_ind
    elif QUANT_T0_MINIMIZE is 'dX':
        ind = dX_min_ind
    elif QUANT_T0_MINIMIZE is 'dP':
        ind = dP_min_ind
    else:
        raise ValueError

    # Minimizing parameters T1, T2, T3, T4:
    # TODO

    # Building a new T grid.
    T1_min_arr[p + 1] = 0
    T1_max_arr[p + 1] = 0
    T4_min_arr[p + 1] = 0
    T4_max_arr[p + 1] = 0

    T2_min_arr[p + 1] = 0
    T2_max_arr[p + 1] = 0
    T3_min_arr[p + 1] = 0
    T3_max_arr[p + 1] = 0