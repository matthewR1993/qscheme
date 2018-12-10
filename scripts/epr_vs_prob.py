import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# det = 'FIRST'
# det = 'THIRD'
# det = 'NONE'
det = 'BOTH'

phase_mod_channel = 1

phase = 0.0

# states_config = 'single(chan-1)_coher(chan-2)'
# states_config = 'single(chan-1)_coher(alpha_0.5_chan-2)'
states_config = 'single(chan-1)_coher(alpha_1_chan-2)'

prob_array = np.linspace(0.005, 0.49, 50)   # 0.622, 0.53

size = len(prob_array)

line = [0.5] * len(prob_array)

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res34/'

epr_x_min_arr = np.zeros(size, dtype=complex)
epr_p_min_arr = np.zeros(size, dtype=complex)

for i in range(size):
    print('step:', i)
    crit_prob = prob_array[i]
    fname = '{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, det, phase_mod_channel)
    fl = np.load(save_root + fname)

    erp_correl_x = fl.item().get('epr_correl_x')
    erp_correl_p = fl.item().get('epr_correl_p')
    prob = fl.item().get('det_prob')

    args_lower = np.argwhere(np.real(prob) < crit_prob)
    for k in range(len(args_lower)):
        index = tuple(args_lower[k, :])
        erp_correl_x[index] = 100
        erp_correl_p[index] = 100

    epr_x_min_arr[i] = np.amin(erp_correl_x)
    epr_p_min_arr[i] = np.amin(erp_correl_p)


plt.plot(prob_array, epr_x_min_arr)
plt.plot(prob_array, line, '-.')
plt.title(f'DET={det}, phase={phase}pi')
plt.xlabel('$Det. probability$')
plt.ylabel('$VAR[X_{1} - X_{2}]$')
plt.grid(True)
plt.show()
