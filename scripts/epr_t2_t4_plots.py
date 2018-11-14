import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt


DET_CONF = 'FIRST'

states_config = 'single(chan-1)_coher(chan-2)'
# states_config = 'coher(chan-1)_single(chan-2)'

phase = 0.0

# crit_prob = 0.1

phase_mod_channel = 1

# phases = [0.25 * n for n in range(8)]
phases = [0]

epr_min_arr = np.zeros(len(phases))

indexes = []

for i, phase in enumerate(phases):
    print('Phase:', phase)
    phase = 0.0
    save_root = '/Users/matvei/PycharmProjects/qscheme/results/res30/'
    # fname = '{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, DET_CONF, phase_mod_channel)
    fname = 'disabled_det_{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, DET_CONF, phase_mod_channel)

    fl = np.load(save_root + fname)

    sqeez_dX = fl.item().get('squeez_dx')
    sqeez_dP = fl.item().get('squeez_dp')
    erp_correl_x = fl.item().get('epr_correl_x')
    erp_correl_p = fl.item().get('epr_correl_p')
    prob = fl.item().get('det_prob')

    t1_arr = fl.item().get('t1_arr')
    t4_arr = fl.item().get('t4_arr')

    # Detection probability.
    min_index = list(np.unravel_index(np.argmin(erp_correl_x, axis=None), erp_correl_x.shape))
    # min_prob = prob[tuple(min_index)]
    # print('min. prob:', min_prob)
    indexes.append(min_index)
    print(min_index)

    # print('EPR_X:', np.amin(np.real(erp_correl_x[0, 0, :, :] / sqrt(1/2))))
    print('EPR_X:', np.amin(np.real(erp_correl_x[:, 0, 0, :])))

    epr_min_arr[i] = np.amin(np.real(erp_correl_x[:, 0, 0, :]))


plt.plot(phases, epr_min_arr)
plt.xlabel('PHASE')
plt.show()

# t1 = [t1_arr[i[0]] for i in indexes]
# t4 = [t1_arr[i[1]] for i in indexes]
#
# plt.plot(phases, t1)
# plt.plot(phases, t4)
# plt.xlabel('PHASE')
# plt.ylabel('t')
# plt.plot()


plt.imshow(np.real(erp_correl_x[:, 0, 0, :]), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
# plt.scatter(x=[min_index[3]], y=[min_index[2]], c='r', s=80, marker='+')
plt.xlabel('T2')
plt.ylabel('T3')
plt.show()

# plt.imshow(np.real(erp_correl_x[:, :, 0, 0] / sqrt(1/2)), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('T1')
# plt.ylabel('T4')
# plt.show()


# Analytics plot of EPR_X.
# a_02 = lambda t, phi: t**2 * sqrt(2) * np.exp(1j*2*phi) - sqrt(2) * (1 - t**2)
# a_11 = lambda t, phi: 1j * 2 * t * sqrt(1 - t**2) * (1 + np.exp(1j*2*phi))
# a_20 = lambda t, phi: (t**2 - 1) * sqrt(2) * np.exp(1j*2*phi) + t**2 * sqrt(2)
#
#
# t_grid = 100
# t_arr = np.linspace(0, 1, t_grid)
#
# epr_x_arr = np.zeros((t_grid), dtype=complex)
#
# phi = 0.5
# for n, t in enumerate(t_arr):
#     epr_x_arr[n] = (1 / 32) * (
#             np.conj(a_02(t, phi)) * (6 * a_02(t, phi) - 2 * sqrt(2) * a_11(t, phi)) +
#             np.conj(a_11(t, phi)) * (- 2*sqrt(2) * a_02(t, phi) + 6 * a_11(t, phi) - 2*sqrt(2) * a_20(t, phi)) +
#             np.conj(a_20(t, phi)) * (- 2*sqrt(2) * a_11(t, phi) + 6 * a_20(t, phi))
#     )
#     print(epr_x_arr[n])
#
#
# plt.plot(t_arr, np.real(epr_x_arr))
# plt.show()


# DET_CONF = 'BOTH'
# phase = 0.0
# phase_mod_channel = 1
# states_config = 'single(chan-1)_coher(chan-2)'
#
# save_root = '/Users/matvei/PycharmProjects/qscheme/results/res27/'
# fname = 'disabled_det_{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, DET_CONF,
#                                                                          phase_mod_channel)
#
# fl = np.load(save_root + fname)
#
# epr_correl_x_4d = fl.item().get('epr_correl_x')
#
#
# epr_correl_x = epr_correl_x_4d[0, :, :, 0]
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))


plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.xlabel('T2')
plt.ylabel('T4')
plt.show()
