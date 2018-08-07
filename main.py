import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from time import gmtime, strftime

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 3
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
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


# The phase difference before last BS
ph_inpi = 1.0
phase_diff = ph_inpi * np.pi

# BS grids.
r1_grid = 11
r4_grid = 11

r2_grid = 1
r3_grid = 1


# BS values range.
T1_min = 0.0
T1_max = 1.0
T4_min = 0.0
T4_max = 1.0

T2_min = sqrt(0.5)
T2_max = sqrt(0.5)
T3_min = sqrt(0.5)
T3_max = sqrt(0.5)

# Varying BSs.
t1_array, r1_array = bs_params(T1_min, T1_max, r4_grid)
t4_array, r4_array = bs_params(T4_min, T4_max, r4_grid)
t2_array, r2_array = bs_params(T2_min, T2_max, r2_grid)
t3_array, r3_array = bs_params(T3_min, T3_max, r3_grid)


log_entropy_subs1_array = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
log_entropy_subs2_array = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
lin_entropy_subs1 = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
lin_entropy_subs2 = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
log_negativity = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
mut_information = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
full_fn_entropy = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
sqeez_dX = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
sqeez_dP = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
erp_correl_x = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)
erp_correl_p = np.zeros((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex)


# Start time.
print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
for n1 in range(r1_grid):
    for n4 in range(r4_grid):
        for n2 in range(r2_grid):
            for n3 in range(r3_grid):
                print('Step [n1, n2, n2, n3]:', n1, n4, n2, n3)
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
                final_dens_matrix = process_all(mut_state_unappl, bs_params, phase_diff=0, det_event='NONE')

                # Trace one channel out of final state
                final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
                print('trace of final reduced matrix 2nd channel:', np.trace(final_traced_subs1))

                # Other channel traced
                final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)
                print('trace of final reduced matrix 4th channel:', np.trace(final_traced_subs2))

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
                print('Log. negativity: ', log_negativity[n1, n4, n2, n3])

                # Squeezing quadratures.
                dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
                print('dX:', dX, ' dP:', dP)
                sqeez_dX[n1, n4, n2, n3] = dX
                sqeez_dP[n1, n4, n2, n3] = dP

                # ERP correlations.
                erp_x, erp_p = erp_squeezing_correlations(final_dens_matrix)
                erp_correl_x[n1, n4, n2, n3] = erp_x
                erp_correl_p[n1, n4, n2, n3] = erp_p
                print('erp_X:', erp_x, ' erp_P:', erp_p)


# Save it.
fl = np.array([log_negativity,
               mut_information,
               sqeez_dX,
               sqeez_dP,
               erp_correl_x,
               erp_correl_p
               ])
save_root = '/Users/matvei/PycharmProjects/qscheme/results/res13/coh(ch1)_single(ch2)_var_phase_t1_t4_det-FIRST/phase-0.5pi/'
# save_root = '/home/matthew/qscheme/results/res13/coh(ch2)_single(ch1)_var_phase_t1_t4_det-FIRST/phase-0.0pi/'
fname = 'phase_0.5pi.npy'
# np.save(save_root + fname, fl)


# fle = np.load(save_root + fname)
# log_negativity = fle[0]
# mut_information = fle[1]
# sqeez_dX = fle[2]
# sqeez_dP = fle[3]
# erp_correl_x = fle[4]
# erp_correl_p = fle[5]


ORTS = [T4_min, T4_max, T1_min, T1_max]


# # 2D pictures
# # Varying t4
# plt.plot(np.square(t4_array), np.real(log_entropy_subs1_array[:, 0]), label=r'$Log. FN \ entropy \ subs \ 1$')
# plt.plot(np.square(t4_array), np.real(log_entropy_subs2_array[:, 0]), label=r'$Log. FN \ entropy \ subs \ 2$')
# plt.plot(np.square(t4_array), np.real(mut_information[:, 0]), label=r'$Mut \ information, \ FN \ log. \ entropy.$')
# plt.plot(np.square(t4_array), np.real(full_fn_entropy[:, 0]), label=r'$Full \ FN \ log. \ entropy.$')
# # plt.plot(np.square(t4_array), np.real(lin_entropy[:, 0]), label=r'$Lin. entropy$')
# plt.plot(np.square(t4_array), np.real(log_negativity[:, 0]), label=r'$Log. \ negativity$')
# # plt.plot(np.square(t4_array), np.real(log_negativity_aftdet[:, 0]), label=r'$Log. negativity, after det. with phase$')
# # plt.title(f'Entanglement. Phase=%0.2f pi. Det. - %s' % (ph_inpi, DET_CONF))
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel('$Entanglement$')
# # plt.title('Phase = {0}pi'.format(ph_inpi))
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # plot squeezing
# plt.plot(np.square(t4_array), 10*np.log10(sqeez_dX[:, 0]/QUADR_VAR_X_VAC), label=r'$10\log_{10}{\frac{\Delta X^{(out)}}{\Delta X^{(vac)}}}$')
# plt.plot(np.square(t4_array), 10*np.log10(sqeez_dP[:, 0]/QUADR_VAR_P_VAC), label=r'$10\log_{10}{\frac{\Delta P^{(out)}}{\Delta P^{(vac)}}}$')
# # plt.plot(np.square(t4_array), np.multiply(sqeez_dX[:, 0], sqeez_dP[:, 0]), label=r'$\Delta X \Delta P$')
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel('$Squeezing$')
# plt.title('$Squeezing$')
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # ERP correlations
# plt.plot(np.square(t4_array), erp_correl_x[:, 0]/EPR_VAR_X_VAC, label=r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$')
# plt.plot(np.square(t4_array), erp_correl_p[:, 0]/EPR_VAR_P_VAC, label=r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$')
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel('$ERP \ correlations \ [a. u.]$')
# plt.title('$ERP \ correlations$')
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # dX
# # dX - 3D - picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# # surf = ax.plot_surface(X, Y, np.real(sqeez_dX), cmap=cm.coolwarm,
# #                         linewidth=0, antialiased=False)
# # plt.title(r'$\Delta X$')
# surf = ax.plot_surface(X, Y, 10*np.log10(np.real(sqeez_dX)/QUADR_VAR_X_VAC), cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$10\log_{10}{(\frac{\Delta X^{(out)}}{\Delta X^{(vac)}})}$', fontsize=16)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# dX - 2D - picture.
# im = plt.imshow(np.real(sqeez_dX), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower', extent=ORTS)
# plt.title('$\Delta X$')
im = plt.imshow(10*np.log10(np.real(sqeez_dX[:, :, 0, 0])/QUADR_VAR_X_VAC), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$10\log_{10}{(\frac{\Delta X^{(out)}}{\Delta X^{(vac)}})}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()
#
#
# # dP - 3D - picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# # surf = ax.plot_surface(X, Y, np.real(sqeez_dP), cmap=cm.coolwarm,
# #                        linewidth=0, antialiased=False)
# # plt.title(r'$\Delta P$')
# surf = ax.plot_surface(X, Y, 10*np.log10(np.real(sqeez_dP)/QUADR_VAR_P_VAC), cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$10*\log_{10}{(\frac{\Delta P^{(out)}}{\Delta P^{(vac)}})}$')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# dP - 2D - picture.
# im = plt.imshow(np.real(sqeez_dP), interpolation='None', cmap=cm.RdYlGn, origin='lower', extent=ORTS)
# plt.title('$\Delta P$')
im = plt.imshow(10*np.log10(np.real(sqeez_dP[:, :, 0, 0])/QUADR_VAR_P_VAC), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$10\log_{10}{(\frac{\Delta P^{(out)}}{\Delta P^{(vac)}})}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()
#
#
# Uncertainty principle: dX * dP >= 1/4.
# Should be >= 1/4
im = plt.imshow(np.multiply(np.real(sqeez_dX[:, :, 0, 0]), np.real(sqeez_dP[:, :, 0, 0])), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\Delta X\Delta P$')
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()
#
#
# # EPR correlations:
# # EPR X, 3D picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# surf = ax.plot_surface(X, Y, np.real(erp_correl_x)/EPR_VAR_X_VAC, cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.xlim(1, 0)
# plt.ylim(0, 1)
# plt.show()
#
#
# # ERP X, 2D picture
im = plt.imshow(np.real(erp_correl_x[:, :, 0, 0])/EPR_VAR_X_VAC, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()

# Trimmed.
size = len(erp_correl_x)
Z_erpX = np.real(erp_correl_x[:, :, 0, 0])/EPR_VAR_X_VAC
for i in range(size):
    for j in range(size):
        if Z_erpX[i, j] > 1:
            Z_erpX[i, j] = 0
im = plt.imshow(Z_erpX, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()



# # ERP P, 3D picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# surf = ax.plot_surface(X, Y, np.real(erp_correl_p)/EPR_VAR_P_VAC, cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.xlim(1, 0)
# plt.ylim(0, 1)
# plt.show()

# ERP P, 2D picture
im = plt.imshow(np.real(erp_correl_p[:, :, 0, 0])/EPR_VAR_P_VAC, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()

# Trimmed.
size = len(erp_correl_p)
Z_erpP = np.real(erp_correl_p[:, :, 0, 0])/EPR_VAR_P_VAC
for i in range(size):
    for j in range(size):
        if Z_erpP[i, j] > 1:
            Z_erpP[i, j] = 0
im = plt.imshow(Z_erpP, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()


# Uncertainty principle for EPR, should be > 1/2???
im = plt.imshow(np.multiply(np.real(erp_correl_x[:, :, 0, 0]), np.real(erp_correl_p[:, :, 0, 0])), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\Delta[X^{(1)} - X^{(2)}]^{(out)}\Delta[P^{(1)} + P^{(2)}]^{(out)}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()
