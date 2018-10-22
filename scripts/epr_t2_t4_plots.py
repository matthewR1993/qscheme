import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt


DET_CONF = 'FIRST'

states_config = 'single(chan-1)_coher(chan-2)'
# states_config = 'coher(chan-1)_single(chan-2)'


phase = 1.5

crit_prob = 0.1

save_root = '/home/matvei/qscheme/results/res26/'
fname = '{}_phase-{:.4f}pi_det-{}.npy'.format(states_config, phase, DET_CONF)

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


print(np.amin(np.real(erp_correl_x[0, :, :, 0] / sqrt(1/2))))

plt.imshow(np.real(erp_correl_x[0, :, :, 0] / sqrt(1/2)), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
# plt.scatter(x=[epr_x_amin_ind[1]], y=[epr_x_amin_ind[0]], c='r', s=80, marker='+')
# plt.scatter(x=[50], y=[50], c='g', s=80, marker='+')
# plt.plot(T1_coord*100, T4_coord*100)
plt.xlabel('T4')
plt.ylabel('T1')
plt.show()
