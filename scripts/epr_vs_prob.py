import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# det = 'FIRST'
det = 'THIRD'
# det = 'NONE'
# det = 'BOTH'

phase_mod_channel = 1

phase = 1.5

# states_config = 'single(chan-1)_coher(chan-2)'
# states_config = 'single(chan-1)_coher(alpha_0.5_chan-2)'
# states_config = 'single(chan-1)_coher(alpha_1_chan-2)'

configs = [
    'single(chan-1)_coher(alpha_0.1_chan-2)',
    'single(chan-1)_coher(alpha_0.2_chan-2)',
    'single(chan-1)_coher(alpha_0.3_chan-2)',
    'single(chan-1)_coher(alpha_0.5_chan-2)',
    'single(chan-1)_coher(alpha_0.75_chan-2)',
    'single(chan-1)_coher(chan-2)'
]

prob_array = np.linspace(0.005, 0.53, 50)   # 0.622, 0.53

size = len(prob_array)

line = [0.5] * len(prob_array)

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res34/'

# epr_x_min_arr = np.zeros(size, dtype=complex)
# epr_p_min_arr = np.zeros(size, dtype=complex)

epr_x_all = []

for j, cnf in enumerate(configs):
    epr_x_min_arr = np.zeros(size, dtype=complex)
    states_config = cnf
    # epr_p_min_arr = np.zeros(size, dtype=complex)
    for i in range(size):
        print('step:', i)
        crit_prob = prob_array[i]
        fname = '{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, det, phase_mod_channel)
        fl = np.load(save_root + fname)

        erp_correl_x = fl.item().get('epr_correl_x')
        # erp_correl_p = fl.item().get('epr_correl_p')
        prob = fl.item().get('det_prob')

        args_lower = np.argwhere(np.real(prob) < crit_prob)
        for k in range(len(args_lower)):
            index = tuple(args_lower[k, :])
            erp_correl_x[index] = 100
            # erp_correl_p[index] = 100

        epr_x_min_arr[i] = np.amin(erp_correl_x)
        # epr_p_min_arr[i] = np.amin(erp_correl_p)
        epr_x_min_arr[epr_x_min_arr > 99.9] = 0


    epr_x_all.append(epr_x_min_arr)

lables = ['alpha=0.1', 'alpha=0.2', 'alpha=0.3', 'alpha=0.5', 'alpha=0.75', 'alpha=1']

for j, cnf in enumerate(configs):
    m = np.trim_zeros(np.real(epr_x_all[j]))
    plt.plot(prob_array[:len(m)], m, label=lables[j])
# plt.plot(prob_array, epr_min_arr, label='alpha=1.2')
plt.plot(prob_array, line, '-.')
plt.title(f'DET={det}, phase={phase}pi')
plt.xlabel('$Det. probability$')
plt.ylabel('$VAR[X_{1} - X_{2}]$')
plt.grid(True)
plt.legend()
plt.show()



# P = 0.1
# P = 0.2
# P = 0.3
alph_arr = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1])
epr_vs_aplha_p0_1 = np.zeros(len(configs))
epr_vs_aplha_p0_2 = np.zeros(len(configs))
epr_vs_aplha_p0_3 = np.zeros(len(configs))
for j, cnf in enumerate(configs):
    m = np.trim_zeros(np.real(epr_x_all[j]))
    epr_vs_aplha_p0_1[j] = m[9]  # 0.1
    epr_vs_aplha_p0_2[j] = m[19]  # 0.2
    epr_vs_aplha_p0_3[j] = m[28]  # 0.3


plt.plot(alph_arr, epr_vs_aplha_p0_1, label='P=0.1')
plt.plot(alph_arr, epr_vs_aplha_p0_2, label='P=0.2')
plt.plot(alph_arr, epr_vs_aplha_p0_3, label='P=0.3')
plt.xlabel('$Alpha$')
plt.ylabel('$VAR[X_{1} - X_{2}]$')
plt.grid(True)
plt.legend()
plt.show()




with open('/Users/matvei/PycharmProjects/qscheme/results/res34/coh_single_DET-F_full.txt') as f:
    content = f.readlines()

cl = [x.strip().split(',') for x in content]

prob_arr = []
epr_arr = []

for c in cl:
    prob = float(c[1])
    prob_arr.append(prob)
    epr = float(c[0])
    epr_arr.append(epr)


epr_arr = np.array(epr_arr)
prob_arr = np.array(prob_arr)


epr_min_arr = np.zeros(len(prob_array))
for j, crt_prob in enumerate(prob_array):
    epr_min = 100
    for i, x in enumerate(epr_arr):
        if prob_arr[i] > crt_prob:
            if epr_arr[i] < epr_min:
                epr_min = epr_arr[i]
    epr_min_arr[j] = epr_min


plt.plot(prob_array, epr_min_arr)
plt.show()
