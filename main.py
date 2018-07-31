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
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!a


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


# Start time.
print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# The first and the last BS grids
r1_grid = 1
r4_grid = 11

bs1_is_symmetrical = True

# The phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

# BS1 transmittion range
T1_min = 0.0
T1_max = 1.0
T4_min = 0.0
T4_max = 1.0

# BS2, BS3
t2 = sqrt(0.9)
r2 = sqrt(1 - t2**2)
t3 = sqrt(0.1)
r3 = sqrt(1 - t3**2)

log_entropy_subs1_array = np.zeros((r4_grid, r1_grid), dtype=complex)
log_entropy_subs2_array = np.zeros((r4_grid, r1_grid), dtype=complex)
lin_entropy_subs1 = np.zeros((r4_grid, r1_grid), dtype=complex)
lin_entropy_subs2 = np.zeros((r4_grid, r1_grid), dtype=complex)
log_negativity = np.zeros((r4_grid, r1_grid), dtype=complex)
mut_information = np.zeros((r4_grid, r1_grid), dtype=complex)
full_fn_entropy = np.zeros((r4_grid, r1_grid), dtype=complex)
sqeez_dX = np.zeros((r4_grid, r1_grid), dtype=complex)
sqeez_dP = np.zeros((r4_grid, r1_grid), dtype=complex)
erp_correl_x = np.zeros((r4_grid, r1_grid), dtype=complex)
erp_correl_p = np.zeros((r4_grid, r1_grid), dtype=complex)


# Varying last BS (BS4)
T4_array = np.linspace(T4_min, T4_max, r4_grid)
t4_array = np.sqrt(T4_array)

r4_fun = lambda tt: sqrt(1 - pow(tt, 2))
r4_vect_func = np.vectorize(r4_fun)
r4_array = r4_vect_func(t4_array)

# Varying first BS (BS1)
T1_array = np.linspace(T1_min, T1_max, r1_grid)
t1_array = np.sqrt(T1_array)

r1_fun = lambda tt: sqrt(1 - pow(tt, 2))
r1_vect_func = np.vectorize(r1_fun)
r1_array = r1_vect_func(t1_array)

# Set BS1 - 50:50 if required.
if r1_grid is 1 and bs1_is_symmetrical:
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
        trim_size = 10
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
        # print('Linear entropy difference: ', lin_entropy_subs1[i, j] - lin_entropy_subs2[i, j])

        log_negativity[i, j] = negativity(final_dens_matrix, neg_type='logarithmic')
        print('Log. negativity: ', log_negativity[i, j])

        # Squeezing quadratures.
        dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
        print('dX:', dX, ' dP:', dP)
        sqeez_dX[i, j] = dX
        sqeez_dP[i, j] = dP

        # ERP correlations.
        erp_x, erp_p = erp_squeezing_correlations(final_dens_matrix)
        erp_correl_x[i, j] = erp_x
        erp_correl_p[i, j] = erp_p
        print('erp_X:', erp_x, ' erp_P:', erp_p)


ORTS = [T4_min, T4_max, T1_min, T1_max]

# 2D pictures
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
plt.plot(np.square(t4_array), 10*np.log10(sqeez_dX[:, 0]/QUADR_VAR_X_VAC), label=r'$10\log_{10}{\frac{\Delta X^{(out)}}{\Delta X^{(vac)}}}$')
plt.plot(np.square(t4_array), 10*np.log10(sqeez_dP[:, 0]/QUADR_VAR_P_VAC), label=r'$10\log_{10}{\frac{\Delta P^{(out)}}{\Delta P^{(vac)}}}$')
#plt.plot(np.square(t4_array), np.multiply(sqeez_dX[:, 0], sqeez_dP[:, 0]), label=r'$\Delta X \Delta P$')
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel('$Squeezing$')
plt.title('$Squeezing$')
plt.xlim([0, 1])
plt.legend()
plt.grid(True)
plt.show()

# ERP correlations
plt.plot(np.square(t4_array), erp_correl_x[:, 0]/EPR_VAR_X_VAC, label=r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$')
plt.plot(np.square(t4_array), erp_correl_p[:, 0]/EPR_VAR_P_VAC, label=r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$')
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel('$ERP \ correlations \ [a. u.]$')
plt.title('$ERP \ correlations$')
plt.xlim([0, 1])
plt.legend()
plt.grid(True)
plt.show()


# dX
# dX - 3D - picture
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.square(t4_array)
Y = np.square(t1_array)
X, Y = np.meshgrid(X, Y)
# surf = ax.plot_surface(X, Y, np.real(sqeez_dX), cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
# plt.title(r'$\Delta X$')
surf = ax.plot_surface(X, Y, 10*np.log10(np.real(sqeez_dX)/QUADR_VAR_X_VAC), cmap=cm.Spectral,
                       linewidth=0, antialiased=False)
plt.title(r'$10\log_{10}{(\frac{\Delta X^{(out)}}{\Delta X^{(vac)}})}$', fontsize=16)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()


# dX - 2D - picture.
# im = plt.imshow(np.real(sqeez_dX), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower', extent=ORTS)
# plt.title('$\Delta X$')
im = plt.imshow(10*np.log10(np.real(sqeez_dX)/QUADR_VAR_X_VAC), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$10\log_{10}{(\frac{\Delta X^{(out)}}{\Delta X^{(vac)}})}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()


# dP - 3D - picture
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.square(t4_array)
Y = np.square(t1_array)
X, Y = np.meshgrid(X, Y)
# surf = ax.plot_surface(X, Y, np.real(sqeez_dP), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.title(r'$\Delta P$')
surf = ax.plot_surface(X, Y, 10*np.log10(np.real(sqeez_dP)/QUADR_VAR_P_VAC), cmap=cm.Spectral,
                       linewidth=0, antialiased=False)
plt.title(r'$10*\log_{10}{(\frac{\Delta P^{(out)}}{\Delta P^{(vac)}})}$')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()


# dP - 2D - picture.
# im = plt.imshow(np.real(sqeez_dP), interpolation='None', cmap=cm.RdYlGn, origin='lower', extent=ORTS)
# plt.title('$\Delta P$')
im = plt.imshow(10*np.log10(np.real(sqeez_dP)/QUADR_VAR_P_VAC), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$10\log_{10}{(\frac{\Delta P^{(out)}}{\Delta P^{(vac)}})}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()


# Uncertainty principle: dX * dP >= 1/4.
# Should be >= 1/4
im = plt.imshow(np.multiply(np.real(sqeez_dX), np.real(sqeez_dP)), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\Delta X\Delta P$')
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()


# EPR correlations:
# EPR X, 3D picture
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.square(t4_array)
Y = np.square(t1_array)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, np.real(erp_correl_x)/EPR_VAR_X_VAC, cmap=cm.Spectral,
                       linewidth=0, antialiased=False)
plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.xlim(1, 0)
plt.ylim(0, 1)
plt.show()

# ERP X, 2D picture
im = plt.imshow(np.real(erp_correl_x)/EPR_VAR_X_VAC, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()

# Trimmed.
size = len(erp_correl_x)
Z_erpX = np.real(erp_correl_x)/EPR_VAR_X_VAC
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


# ERP P, 3D picture
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.square(t4_array)
Y = np.square(t1_array)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, np.real(erp_correl_p)/EPR_VAR_P_VAC, cmap=cm.Spectral,
                       linewidth=0, antialiased=False)
plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.xlim(1, 0)
plt.ylim(0, 1)
plt.show()

# ERP P, 2D picture
im = plt.imshow(np.real(erp_correl_p)/EPR_VAR_P_VAC, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()

# Trimmed.
size = len(erp_correl_p)
Z_erpP = np.real(erp_correl_p)/EPR_VAR_P_VAC
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


# Uncertainty principle for EPR, should be > 1/2
im = plt.imshow(np.multiply(np.real(erp_correl_x), np.real(erp_correl_p)), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
plt.title(r'$\Delta[X^{(1)} - X^{(2)}]^{(out)}\Delta[P^{(1)} + P^{(2)}]^{(out)}$', fontsize=16)
plt.colorbar(im)
plt.xlabel(r'$T_{4}$', fontsize=16)
plt.ylabel(r'$T_{1}$', fontsize=16)
plt.show()
