# Usage: python3 grid_search.py --det FIRST --phase 0.0 --quant EPR_X
import numpy as np
from time import gmtime, strftime
import sys
import platform
import argparse

if platform.system() == 'Linux':
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
    sys.path.append('/home/matthew/qscheme')
elif platform.system() == 'Darwin':
    sys.path.append('/Users/matvei/PycharmProjects/qscheme')

from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--det", help="Detection", type=str, required=True)
parser.add_argument("-p", "--phase", help="Phase in pi", type=float, required=True)
parser.add_argument("-q", "--quant", help="A quantity to minimize", type=str, required=True)
args = parser.parse_args()

source_root = '/Users/matvei/PycharmProjects/qscheme/results/res19_rough/'
# source_root = '/home/matthew/qscheme/results/res19_rough/'
source_fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(args.phase, args.det)
print('Source file path:', source_root + source_fname)

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res19_incr_accuracy/'
# save_root = '/home/matthew/qscheme/results/res19_incr_accuracy/'
save_fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}_quant-{}.npy'.format(args.phase, args.det, args.quant)
print('Saving path:', save_root + save_fname)

phase_diff = args.phase
DET_CONF = args.det

crit_probability = 0.1

# Find minimum
fl = np.load(source_root + source_fname)

sqeez_dX = fl.item().get('squeez_dx')
sqeez_dP = fl.item().get('squeez_dp')
erp_correl_x = fl.item().get('epr_correl_x')
erp_correl_p = fl.item().get('epr_correl_p')
prob = fl.item().get('det_prob')

t1_arr = fl.item().get('t1_arr')
t4_arr = fl.item().get('t4_arr')
t2_arr = fl.item().get('t2_arr')
t3_arr = fl.item().get('t3_arr')

T1_arr = np.square(t1_arr)
T4_arr = np.square(t4_arr)
T2_arr = np.square(t2_arr)
T3_arr = np.square(t3_arr)

delta_T1 = T1_arr[1] - T1_arr[0]
delta_T4 = T4_arr[1] - T4_arr[0]
delta_T2 = T2_arr[1] - T2_arr[0]
delta_T3 = T3_arr[1] - T3_arr[0]

print('Delta T1, T4, T2, T3:', delta_T1, delta_T4, delta_T2, delta_T3)

prob_args_lower = np.argwhere(np.real(prob) < crit_probability)
for i in range(len(prob_args_lower)):
    index = tuple(prob_args_lower[i, :])
    erp_correl_x[index] = 100
    erp_correl_p[index] = 100
    sqeez_dX[index] = 100
    sqeez_dP[index] = 100

# Minimizing indexes.
dX_min_ind = list(np.unravel_index(np.argmin(sqeez_dX, axis=None), sqeez_dX.shape))
dP_min_ind = list(np.unravel_index(np.argmin(sqeez_dP, axis=None), sqeez_dP.shape))
epr_x_min_ind = list(np.unravel_index(np.argmin(erp_correl_x, axis=None), erp_correl_x.shape))
epr_p_min_ind = list(np.unravel_index(np.argmin(erp_correl_p, axis=None), erp_correl_p.shape))

# Minimizing T coordinates.
dX_min_ind_T_arr = np.array([T1_arr[dX_min_ind[0]], T4_arr[dX_min_ind[1]], T2_arr[dX_min_ind[2]], T3_arr[dX_min_ind[3]]])
dP_min_ind_T_arr = np.array([T1_arr[dP_min_ind[0]], T4_arr[dP_min_ind[1]], T2_arr[dP_min_ind[2]], T3_arr[dP_min_ind[3]]])
epr_x_min_T_arr = np.array([T1_arr[epr_x_min_ind[0]], T4_arr[epr_x_min_ind[1]], T2_arr[epr_x_min_ind[2]], T3_arr[epr_x_min_ind[3]]])
epr_p_min_T_arr = np.array([T1_arr[epr_p_min_ind[0]], T4_arr[epr_p_min_ind[1]], T2_arr[epr_p_min_ind[2]], T3_arr[epr_p_min_ind[3]]])

# Minimizing t coordinates.
dX_min_ind_t_arr = np.array([t1_arr[dX_min_ind[0]], t4_arr[dX_min_ind[1]], t2_arr[dX_min_ind[2]], t3_arr[dX_min_ind[3]]])
dP_min_ind_t_arr = np.array([t1_arr[dP_min_ind[0]], t4_arr[dP_min_ind[1]], t2_arr[dP_min_ind[2]], t3_arr[dP_min_ind[3]]])
epr_x_min_t_arr = np.array([t1_arr[epr_x_min_ind[0]], t4_arr[epr_x_min_ind[1]], t2_arr[epr_x_min_ind[2]], t3_arr[epr_x_min_ind[3]]])
epr_p_min_t_arr = np.array([t1_arr[epr_p_min_ind[0]], t4_arr[epr_p_min_ind[1]], t2_arr[epr_p_min_ind[2]], t3_arr[epr_p_min_ind[3]]])


# Building a new coordinate grid around minimum point.
grd_mut = 11

min_quantity = args.quant
print('Quantity to miminize:', min_quantity)

# A new grid's center.
if min_quantity == 'EPR_X':
    min_T_coord = epr_x_min_T_arr
elif min_quantity == 'EPR_P':
    min_T_coord = epr_p_min_T_arr
elif min_quantity == 'QUADR_X':
    min_T_coord = dX_min_ind_T_arr
elif min_quantity == 'QUADR_P':
    min_T_coord = dP_min_ind_T_arr
else:
    raise ValueError

print('Min. T values from the previous step [T1, T4, T2, T3]:', min_T_coord)


delta = 0.1

min_bound = 1e-5
max_bound = 1 - 1e-5


T1_new_max = min_T_coord[0] + delta
T1_new_min = min_T_coord[0] - delta
T4_new_max = min_T_coord[1] + delta
T4_new_min = min_T_coord[1] - delta

T2_new_max = min_T_coord[2] + delta
T2_new_min = min_T_coord[2] - delta
T3_new_max = min_T_coord[3] + delta
T3_new_min = min_T_coord[3] - delta

# Satisfy boundary conditions.
if T1_new_max >= 1:
    T1_new_max = 1
if T1_new_min <= 0:
    T1_new_min = 0
if T4_new_max >= 1:
    T4_new_max = 1
if T4_new_min <= 0:
    T4_new_min = 0

if T2_new_max >= max_bound:
    T2_new_max = max_bound
if T2_new_min <= min_bound:
    T2_new_min = min_bound
if T3_new_max >= max_bound:
    T3_new_max = max_bound
if T3_new_min <= min_bound:
    T3_new_min = min_bound


print('New T1_min, T1_max:', T1_new_min, T1_new_max)
print('New T4_min, T4_max:', T4_new_min, T4_new_max)
print('New T2_min, T2_max:', T2_new_min, T2_new_max)
print('New T3_min, T3_max:', T3_new_min, T3_new_max)

t1_array, _ = bs_parameters(T1_new_min, T1_new_max, grd_mut)
t4_array, _ = bs_parameters(T4_new_min, T4_new_max, grd_mut)
t2_array, _ = bs_parameters(T2_new_min, T2_new_max, grd_mut)
t3_array, _ = bs_parameters(T3_new_min, T3_new_max, grd_mut)

# Adding previous values
t1_array = np.append(t1_array, np.sqrt(min_T_coord)[0])
t4_array = np.append(t4_array, np.sqrt(min_T_coord)[1])
t2_array = np.append(t2_array, np.sqrt(min_T_coord)[2])
t3_array = np.append(t3_array, np.sqrt(min_T_coord)[3])


print("New t1 array:", t1_array)
print("New t4 array:", t4_array)
print("New t2 array:", t2_array)
print("New t3 array:", t3_array)

print("New T1 array:", np.square(t1_array))
print("New T4 array:", np.square(t4_array))
print("New T2 array:", np.square(t2_array))
print("New T3 array:", np.square(t3_array))


# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# INPUT - the state in the first(at the bottom) channel
input_st = coherent_state(input_series_length, alpha=1)

# AUXILIARY - the state in the second(on top) channel
auxiliary_st = single_photon(series_length)

mut_state_unappl = np.tensordot(input_st, auxiliary_st, axes=0)

# Building a new coordinate grid around minimum point.
grd_mut = 11

det_prob_array = np.zeros((grd_mut + 1,)*4, dtype=complex)
log_entropy_subs1_array = np.zeros((grd_mut + 1,)*4, dtype=complex)
log_entropy_subs2_array = np.zeros((grd_mut + 1,)*4, dtype=complex)
lin_entropy_subs1 = np.zeros((grd_mut + 1,)*4, dtype=complex)
lin_entropy_subs2 = np.zeros((grd_mut + 1,)*4, dtype=complex)
log_negativity = np.zeros((grd_mut + 1,)*4, dtype=complex)
mut_information = np.zeros((grd_mut + 1,)*4, dtype=complex)
full_fn_entropy = np.zeros((grd_mut + 1,)*4, dtype=complex)
sqeez_dX = np.zeros((grd_mut + 1,)*4, dtype=complex)
sqeez_dP = np.zeros((grd_mut + 1,)*4, dtype=complex)
epr_correl_x = np.zeros((grd_mut + 1,)*4, dtype=complex)
epr_correl_p = np.zeros((grd_mut + 1,)*4, dtype=complex)
norm_after_det_arr = np.zeros((grd_mut + 1,)*4, dtype=complex)
final_dens_matrix_list = []


if __name__ == "__main__":
    # Start time.
    print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    for n1 in range(grd_mut + 1):
        for n4 in range(grd_mut + 1):
            for n2 in range(grd_mut + 1):
                for n3 in range(grd_mut + 1):
                    print('Steps [n1, n4, n2, n3]:', n1, n4, n2, n3)
                    bs_params = {
                        't1': t1_array[n1],
                        't4': t4_array[n4],
                        't2': t2_array[n2],
                        't3': t3_array[n3],
                    }

                    final_dens_matrix, det_prob, norm = process_all(mut_state_unappl, bs_params, phase_diff=phase_diff, det_event=DET_CONF)
                    if final_dens_matrix is None or det_prob is None:
                        print('Warning: the norm is zero.')
                        pass

                    det_prob_array[n1, n4, n2, n3] = det_prob
                    norm_after_det_arr[n1, n4, n2, n3] = norm
                    # final_dens_matrix_list.append({'dm': final_dens_matrix, 'keys': [n1, n4, n2, n3]})

                    # Trace one channel out of final state
                    # final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
                    # print('trace of final reduced matrix 2nd channel:', np.trace(final_traced_subs1))

                    # Other channel traced
                    # final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)
                    # print('trace of final reduced matrix 4th channel:', np.trace(final_traced_subs2))

                    # Calculate entropy
                    # log_entanglement_subs1 = log_entropy(final_traced_subs1)
                    # log_entanglement_subs2 = log_entropy(final_traced_subs2)
                    # log_entropy_subs1_array[n1, n4, n2, n3] = log_entanglement_subs1
                    # log_entropy_subs2_array[n1, n4, n2, n3] = log_entanglement_subs2

                    # Full entropy and the mutual information
                    # final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
                    # full_entr = log_entropy(final_reorg_matr)

                    # mut_information[n1, n4, n2, n3] = log_entanglement_subs1 + log_entanglement_subs2 - full_entr
                    # full_fn_entropy[n1, n4, n2, n3] = full_entr

                    log_negativity[n1, n4, n2, n3] = negativity(final_dens_matrix, neg_type='logarithmic')
                    # print('Log. negativity: ', log_negativity[n1, n4, n2, n3])

                    # Squeezing quadratures.
                    dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
                    # print('dX:', dX, ' dP:', dP)
                    sqeez_dX[n1, n4, n2, n3] = dX
                    sqeez_dP[n1, n4, n2, n3] = dP

                    # ERP correlations.
                    epr_x, epr_p = erp_squeezing_correlations(final_dens_matrix)
                    epr_correl_x[n1, n4, n2, n3] = epr_x
                    epr_correl_p[n1, n4, n2, n3] = epr_p
                    # print('erp_X:', erp_x, ' erp_P:', erp_p)

    # Save it.
    fl = {
        'det_prob': det_prob_array,
        'norm_aft_det': norm_after_det_arr,
        # 'final_dens_matrix': final_dens_matrix_list,
        'log_negativity': log_negativity,
        # 'mut_inform': mut_information,
        'squeez_dx': sqeez_dX,
        'squeez_dp': sqeez_dP,
        'epr_correl_x': epr_correl_x,
        'epr_correl_p': epr_correl_p,
        'det_conf': args.det,
        'phase': args.phase,
        't1_arr': t1_array,
        't4_arr': t4_array,
        't2_arr': t2_array,
        't3_arr': t3_array,
        'states_config': 'coh(chan-1)_single(chan-2)'
    }
    np.save(save_root + save_fname, fl)


