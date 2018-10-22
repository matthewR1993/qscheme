import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt


DET_CONF = 'FIRST'

# states_config = 'single(chan-1)_coher(chan-2)'
states_config = 'coher(chan-1)_single(chan-2)'


phase = 1.5

# crit_prob = 0.1

phase_mod_channel = 2

save_root = '/home/matvei/qscheme/results/res26/'
fname = '{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, DET_CONF, phase_mod_channel)

fl = np.load(save_root + fname)

sqeez_dX = fl.item().get('squeez_dx')
sqeez_dP = fl.item().get('squeez_dp')
erp_correl_x = fl.item().get('epr_correl_x')
erp_correl_p = fl.item().get('epr_correl_p')
prob = fl.item().get('det_prob')


# TODO filter lower probability.
# args_lower = np.argwhere(np.real(prob) < crit_prob)
# for k in range(len(args_lower)):
#     index = tuple(args_lower[k, :])
#     sqeez_dX[index] = 100
#     sqeez_dP[index] = 100
#     erp_correl_x[index] = 100
#     erp_correl_p[index] = 100


# Detection probability.
min_index = list(np.unravel_index(np.argmin(erp_correl_x, axis=None), erp_correl_x.shape))
min_prob = prob[tuple(min_index)]
print('min. prob:', min_prob)

print('EPR_X:', np.amin(np.real(erp_correl_x[0, 0, :, :] / sqrt(1/2))))

plt.imshow(np.real(erp_correl_x[0, 0, :, :] / sqrt(1/2)), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
# plt.scatter(x=[epr_x_amin_ind[1]], y=[epr_x_amin_ind[0]], c='r', s=80, marker='+')
plt.xlabel('T2')
plt.ylabel('T3')
plt.show()


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
