# Measuring performance.

import sys
import time
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

from time import gmtime, strftime
from numpy.testing import assert_array_equal, assert_allclose

from customutils.utils import *
from core.basic import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from setup_parameters import *
from core.optimized import transformations as trans


# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
# input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel
# auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

mut_state_unappl = np.tensordot(input_st, auxiliary_st, axes=0)


# The phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

# BS2, BS3.
t1 = sqrt(0.5)
r1 = sqrt(1 - t1**2)
t4 = sqrt(0.5)
r4 = sqrt(1 - t4**2)

t2 = sqrt(0.5)
r2 = sqrt(1 - t2**2)
t3 = sqrt(0.5)
r3 = sqrt(1 - t3**2)

# Measurements start here.
start1 = time.time()

# First BS.
start = time.time()
state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)
end = time.time()
print('First BS time:', end - start)


# 2d and 3rd BS.
trm = 12
start = time.time()
# state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl[:trm, :trm])
state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)
end = time.time()
print('BS 2 and 3 time:', end - start)

# np.sum(state_after_bs_unappl[:trm, :trm]) - np.sum(state_after_bs_unappl)

#
# def two_bs2x4_transform_opt(t1, r1, t2, r2, input_state):
#     size = len(input_state)
#     output_state = np.zeros((size,) * 4, dtype=complex)
#
#     def coef(k1, k2, k3, k4): return t1 ** (k2) * (1j * r1) ** k1 * t2 ** (k4) * (1j * r2) ** k3 / (factorial(k1) * factorial(k2) * factorial(k3) * factorial(k4))
#
#     # index 'i' => (m,n,k,l)
#     for i in np.ndindex(size, size, size, size):
#         if i[2] <= i[0] and i[3] <= i[1] and i[0] + i[1] < size:
#             output_state[i[2], i[0] - i[2], i[3], i[1] - i[3]] = coef(i[2], i[0] - i[2], i[3], i[1] - i[3]) * input_state[i[0], i[1]] * factorial(i[0]) * factorial(i[1])
#
#     return output_state


start = time.time()
state_aft2bs_unappl_opt = two_bs2x4_transform_opt(t2, r2, t3, r3, state_after_bs_unappl)
# state_aft2bs_unappl_opt = two_bs2x4_transform_opt(t2, r2, t3, r3, state_after_bs_unappl[:trm, :trm])
end = time.time()
print('BS 2 and 3 time OPT:', end - start)

assert_allclose(state_aft2bs_unappl_opt, state_aft2bs_unappl)

print(np.sum(state_aft2bs_unappl_opt - state_aft2bs_unappl))


print(t1, r1, t2, r2)

state2 = np.tensordot(fock_state(2, series_length), fock_state(2, series_length), axes=0)
out_state2 = np.zeros((series_length,) * 4, dtype=complex)
out_state2[0, 2, 0, 2] = t1 ** 2 * t2 ** 2
out_state2[0, 2, 1, 1] = 2j * t1 ** 2 * t2 * r2
out_state2[0, 2, 2, 0] = - t1 ** 2 * r2 ** 2
out_state2[1, 1, 0, 2] = 2j * t1 * r1 * t2 ** 2
out_state2[1, 1, 1, 1] = - 4 * t1 * r1 * t2 * r2
out_state2[1, 1, 2, 0] = - 2j * t1 * r1 * r2 ** 2
out_state2[2, 0, 0, 2] = - r1 ** 2 * t2 ** 2
out_state2[2, 0, 1, 1] = - 2j * r1 ** 2 * t2 * r2
out_state2[2, 0, 2, 0] = r1 ** 2 * r2 ** 2
out_state2 = out_state2 * 0.5  # initial state is unapplied

state_aft2bs_unappl_opt = two_bs2x4_transform_opt(t1, r1, t2, r2, state2)
state_aft2bs_unappl = two_bs2x4_transform(t1, r1, t2, r2, state2)

print(np.sum(state_aft2bs_unappl_opt - out_state2))
print(np.sum(state_aft2bs_unappl - out_state2))

# for i in np.ndindex(size, size, size, size):
#     k2 = i[0] - i[2]
#     k4 = i[1] - i[3]
#     if i[2] <= i[0] and i[3] <= i[1] and i[0] + i[1] < size:
#         output_state[i[2], k2, i[3], k4] = input_state[i[0], i[1]] * factorial(i[0]) * factorial(i[1]) * t1 ** (k2) * (
#                     1j * r1) ** i[2] * t2 ** (k4) * (1j * r2) ** i[3] / (
#                                                    factorial(i[2]) * factorial(k2) * factorial(i[3]) * factorial(k4))

start = time.time()
det_prob = det_probability(state_aft2bs_unappl, detection_event=DET_CONF)
end = time.time()
print('Det prob. time:', end - start)

# The detection event.
start = time.time()
# Gives non-normalised state.
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
end = time.time()
print('Detection:', end - start)
# Calculating the norm.
# start = time.time()
# norm_after_det = state_norm(state_after_dett_unappl)
# end = time.time()
# print('Calc norm after det:', end - start)
# print('Norm after det.:', norm_after_det)
# The normalised state.

# New norm
start = time.time()
norm_after_det_new = state_norm_opt(state_after_dett_unappl)
end = time.time()
print('State norm after det NEW:', end - start, '\n')
# print(norm_after_det - norm_after_det_new)

state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det_new


trim_st = 8
state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:trim_st, :trim_st, :trim_st, :trim_st]

# Trimmed! 2 sec.
# start = time.time()
# dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)
# end = time.time()
#print('Dens. matrix with trace, TRIMMED:', end - start, '\n')

start = time.time()
dens_matrix_2channels_opt = dens_matrix_with_trace_opt(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)
end = time.time()
print('Dens. matrix with trace, OPT:', end - start, '\n')

# print('Diff', np.sum(dens_matrix_2channels - dens_matrix_2channels_opt))


# Phase modulation.
start = time.time()
dens_matrix_2channels_withph = phase_modulation(dens_matrix_2channels_opt, phase_diff)
end = time.time()
print('Phase modulation:', end - start)


# Dens matrix BS transform.
trim_size = 7
start = time.time()
final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels_withph[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)
end = time.time()
print('BS4 density matrix transformation:', end - start)


start = time.time()
final_dens_matrix_new = trans.bs_matrix_transform_opt(dens_matrix_2channels_withph[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)
end = time.time()
print('BS4 density matrix transformation NEW:', end - start)

# Comparing difference of matrices
print(np.sum(final_dens_matrix - final_dens_matrix_new))

# print(np.sum(final_dens_matrix) - np.sum(final_dens_matrix[:10, :10, :10, :10]))


# Todo, another optimisation.
def bs_densmatrix_transform_opt2(input_matrix, t, r):
    size = len(input_matrix)
    output_matrix = np.zeros((size*2,) * 4, dtype=complex)

    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):

                    # four sums
                    for n in range(p1 + 1):
                        for k in range(p2 + 1):
                            for n_ in range(p1_ + 1):
                                for k_ in range(p2_ + 1):
                                    d1 = p1 - n + k
                                    d2 = n + p2 - k
                                    coeff1 = t**(p1 - n + p2 - k) * (1j*r)**(n + k) * sqrt(factorial(d1) * factorial(d2) * factorial(p1) * factorial(p2)) / (factorial(n) * factorial(p1 - n) * factorial(k) * factorial(p2 - k))

                                    d1_ = p1_ - n_ + k_
                                    d2_ = n_ + p2_ - k_
                                    coeff2 = t**(p1_ - n_ + p2_ - k_) * (-1j*r)**(n_ + k_) * sqrt(factorial(d1_) * factorial(d2_) * factorial(p1_) * factorial(p2_)) / (factorial(n_) * factorial(p1_ - n_) * factorial(k_) * factorial(p2_ - k_))

                                    output_matrix[d1, d2, d1_, d2_] = output_matrix[d1, d2, d1_, d2_] + input_matrix[p1, p2, p1_, p2_] * coeff1 * coeff2

    return output_matrix


# end1 = time.time()
# print('Overall:', end1 - start1)
#
#
# start = time.time()
# final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
# end = time.time()
# print('Trace one channel out of the final ds. matrix:', end - start)
#
# start = time.time()
# log_entanglement_subs1 = log_entropy(final_traced_subs1)
# end = time.time()
# print('Log. entropy:', end - start)
#
# start = time.time()
# final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
# full_entr = log_entropy(final_reorg_matr)
# end = time.time()
# print('Full entropy:', end - start)
#
# start = time.time()
# negativity(final_dens_matrix, neg_type='logarithmic')
# end = time.time()
# print('Negativity:', end - start)
#
# start = time.time()
# # Squeezing quadratures.
# dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
# end = time.time()
# print('Squezing quadratures:', end - start)
#
# start = time.time()
# # ERP correlations.
# erp_x, erp_p = erp_squeezing_correlations(final_dens_matrix)
# end = time.time()
# print('EPR correlations:', end - start)
#
# end1 = time.time()
# print('Overall:', end1 - start1)


# import numpy as np
# from core.optimized import transformations
# dm = np.zeros((10,) * 4, dtype=complex)
# cc = transformations.bs_matrix_transform_opt(dm, 0.37, 0.57)
# print(cc[0, 0, 0, 0])

