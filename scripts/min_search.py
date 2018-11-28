import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt


r1_grid = 11
r4_grid = 11
r2_grid = 11
r3_grid = 11

# det = 'FIRST'
det = 'THIRD'
# det = 'NONE'
# det = 'BOTH'

phase_mod_channel = 1

states_config = 'single(chan-1)_coher(chan-2)'

# phases = [x * 0.25 for x in range(9)]
phases = [x * 0.125 for x in range(17)]
# phases = [0.25]
# phases = [x * 0.25 for x in range(5)]

crit_prob = 0.1

size = len(phases)

line = [0.5] * len(phases)

neg_max_arr = np.zeros(size, dtype=complex)
dX_min_arr = np.zeros(size, dtype=complex)
dP_min_arr = np.zeros(size, dtype=complex)
epr_x_min_arr = np.zeros(size, dtype=complex)
epr_p_min_arr = np.zeros(size, dtype=complex)
uncert_min_arr = np.zeros(size, dtype=complex)
epr_x_min_prob_arr = np.zeros(size, dtype=complex)
epr_p_min_prob_arr = np.zeros(size, dtype=complex)
dX_min_prob_arr = np.zeros(size, dtype=complex)
dP_min_prob_arr = np.zeros(size, dtype=complex)

dX_min_ind = np.zeros(size, dtype=list)
dP_min_ind = np.zeros(size, dtype=list)
epr_x_min_ind = np.zeros(size, dtype=list)
epr_p_min_ind = np.zeros(size, dtype=list)
uncert_min_ind = np.zeros(size, dtype=list)

t1_arr_list = np.zeros(size, dtype=list)
t4_arr_list = np.zeros(size, dtype=list)
t2_arr_list = np.zeros(size, dtype=list)
t3_arr_list = np.zeros(size, dtype=list)


for i in range(size):
    print('step:', i)
    phase = phases[i]

    save_root = '/Users/matvei/PycharmProjects/qscheme/results/res28/'
    # save_root = '/home/matvei/qscheme/results/res31/'
    fname = '{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, phase, det, phase_mod_channel)

    fl = np.load(save_root + fname)

    sqeez_dX = fl.item().get('squeez_dx')
    sqeez_dP = fl.item().get('squeez_dp')
    erp_correl_x = fl.item().get('epr_correl_x')
    erp_correl_p = fl.item().get('epr_correl_p')
    prob = fl.item().get('det_prob')
    neg = fl.item().get('log_negativity')

    args_lower = np.argwhere(np.real(prob) < crit_prob)
    for k in range(len(args_lower)):
        index = tuple(args_lower[k, :])
        sqeez_dX[index] = 100
        sqeez_dP[index] = 100
        erp_correl_x[index] = 100
        erp_correl_p[index] = 100

    uncert_min_arr[i] = np.amin(np.multiply(sqeez_dX, sqeez_dP))
    dX_min_arr[i] = np.amin(sqeez_dX)
    dP_min_arr[i] = np.amin(sqeez_dP)
    epr_x_min_arr[i] = np.amin(erp_correl_x)
    epr_p_min_arr[i] = np.amin(erp_correl_p)
    neg_max_arr[i] = np.amax(neg)

    uncert = np.multiply(sqeez_dX, sqeez_dP)

    uncert_min_ind[i] = list(np.unravel_index(np.argmin(uncert, axis=None), uncert.shape))
    dX_min_ind[i] = list(np.unravel_index(np.argmin(sqeez_dX, axis=None), sqeez_dX.shape))
    dP_min_ind[i] = list(np.unravel_index(np.argmin(sqeez_dP, axis=None), sqeez_dP.shape))
    epr_x_min_ind[i] = list(np.unravel_index(np.argmin(erp_correl_x, axis=None), erp_correl_x.shape))
    epr_p_min_ind[i] = list(np.unravel_index(np.argmin(erp_correl_p, axis=None), erp_correl_p.shape))

    epr_x_min_prob_arr[i] = prob[tuple(epr_x_min_ind[i])]
    epr_p_min_prob_arr[i] = prob[tuple(epr_p_min_ind[i])]
    dX_min_prob_arr[i] = prob[tuple(dX_min_ind[i])]
    dP_min_prob_arr[i] = prob[tuple(dP_min_ind[i])]

    t1_arr_list[i] = fl.item().get('t1_arr')
    t4_arr_list[i] = fl.item().get('t4_arr')
    t2_arr_list[i] = fl.item().get('t2_arr')
    t3_arr_list[i] = fl.item().get('t3_arr')


# Negativity.
# plt.plot(phases, neg_max_arr, 'r.')
# plt.show()


# Uncertainty. Should be 1/4.
# plt.plot(phases, uncert_min_arr, 'r.')
# plt.title('$dPdX^{min}$')
# plt.ylabel('$dPdX^min}$')
# plt.xlabel('$phase \ in \ \pi$')
# plt.show()

# # Quadratures.
# plt.plot(phases, 10*np.log10(dX_min_arr/QUADR_VAR_X_VAC), 'r-o')
# plt.title(r'$10\log_{10}{\frac{\Delta X^{(out)}}{\Delta X^{(vac)}}}$')
# plt.xlabel('$Phase, [\pi]$')
# plt.grid(True)
# plt.show()
#
# plt.plot(phases, 10*np.log10(dP_min_arr/QUADR_VAR_P_VAC), 'b-o')
# plt.title(r'$10\log_{10}{\frac{\Delta P^{(out)}}{\Delta P^{(vac)}}}$')
# plt.xlabel('$Phase, [\pi]$')
# plt.grid(True)
# plt.show()

# EPR:
plt.plot(phases, epr_x_min_arr, 'r-o')
plt.title(r'$VAR[X^{(1)} - X^{(2)}]^{(out)}$', fontsize=18)
plt.plot(phases, line, '-.')
plt.xlabel('$Phase, [\pi]$', fontsize=18)
plt.grid(True)
plt.xlim(0, 2)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()


plt.plot(phases, epr_p_min_arr, 'b-o')
plt.title(r'$VAR[P^{(1)} + P^{(2)}]^{(out)}$')
plt.plot(phases, line, '-.')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.xlim(0, 2)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()


# Probabilities of realisation for EPR:
plt.plot(phases, epr_x_min_prob_arr, 'r-o')
plt.title('$Prob[EPR[X]^{min}]$')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.show()

plt.plot(phases, epr_p_min_prob_arr, 'r-o')
plt.title('$Prob[EPR[P]^{min}]$')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.show()


# Minimizing 'T' points for each phase.
epr_x_min_T_arr = np.zeros((size, 4), dtype=float)
epr_p_min_T_arr = np.zeros((size, 4), dtype=float)
dX_min_T_arr = np.zeros((size, 4), dtype=float)
dP_min_T_arr = np.zeros((size, 4), dtype=float)
for j in range(size):
    T1_arr = np.square(t1_arr_list[j])
    T4_arr = np.square(t4_arr_list[j])
    T2_arr = np.square(t2_arr_list[j])
    T3_arr = np.square(t3_arr_list[j])
    epr_x_min_T_arr[j, :] = np.array([T1_arr[epr_x_min_ind[j][0]], T4_arr[epr_x_min_ind[j][1]], T2_arr[epr_x_min_ind[j][2]], T3_arr[epr_x_min_ind[j][3]]])


df_phase = pd.DataFrame(np.array(phases))

df_dX_min_ind = pd.concat([df_phase, pd.DataFrame(np.real(dX_min_prob_arr)), pd.DataFrame(dX_min_T_arr)], axis=1, ignore_index=True)
df_dX_min_ind.columns = ['Phase, [pi]', 'Probab.', 'T1', 'T4', 'T2', 'T3']
df_dP_min_ind = pd.concat([df_phase, pd.DataFrame(np.real(dP_min_prob_arr)), pd.DataFrame(dP_min_T_arr)], axis=1, ignore_index=True)
df_dP_min_ind.columns = ['Phase, [pi]', 'Probab.', 'T1', 'T4', 'T2', 'T3']
df_epr_x_min_ind = pd.concat([df_phase, pd.DataFrame(np.real(epr_x_min_prob_arr)), pd.DataFrame(epr_x_min_T_arr)], axis=1, ignore_index=True)
df_epr_x_min_ind.columns = ['Phase, [pi]', 'Probab.', 'T1', 'T4', 'T2', 'T3']
df_epr_p_min_ind = pd.concat([df_phase, pd.DataFrame(np.real(epr_p_min_prob_arr)), pd.DataFrame(epr_p_min_T_arr)], axis=1, ignore_index=True)
df_epr_p_min_ind.columns = ['Phase, [pi]', 'Probab.', 'T1', 'T4', 'T2', 'T3']


# Plot tables.
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
# df = df_dX_min_ind
# df = df_dP_min_ind
df = df_epr_x_min_ind.round(4)
# df = df_epr_p_min_ind
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
fig.tight_layout()
plt.show()


# Load the result from theory.
save_root = '/Users/matvei/PycharmProjects/qscheme/results/res28/'
# save_root = '/home/matvei/qscheme/results/res28/'
fname = 'epr_x_min_vs_phase_theory.npy'

fl = np.load(save_root + fname)

epr_x_min_arr_th = fl.item().get('epr_x_min')
phase_arr_th = fl.item().get('phases')
min_indexes = fl.item().get('min_index')

plt.plot(phase_arr_th / np.pi, epr_x_min_arr_th)
plt.xlabel('$Phase, [\pi]$')
plt.plot(phase_arr_th / np.pi, [0.5]*len(phase_arr_th), '-.')
plt.grid(True)
plt.show()

# Theory and numer. together.
plt.plot(phases, epr_x_min_arr, 'r-o', label='With detection.')
plt.title(r'$VAR[X^{(1)} - X^{(2)}]^{(out)}$', fontsize=18)
plt.plot(phases, line, '-.')
plt.plot(phase_arr_th / np.pi, epr_x_min_arr_th, label='Without detection.')
plt.xlabel('$Phase, [\pi]$', fontsize=18)
plt.grid(True)
plt.xlim(0, 2)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend()
plt.show()


# Plot parameters.
t_grd = 100
t1_arr = np.linspace(0, 1, t_grd)
t2_arr = np.linspace(0, 1, t_grd)

t1_min_vals = np.zeros(len(min_indexes))
t2_min_vals = np.zeros(len(min_indexes))


for i, item in enumerate(min_indexes):
    print(item)
    t1_min_vals[i] = t1_arr[item[0]]
    t2_min_vals[i] = t2_arr[item[1]]


plt.plot(phase_arr_th, t1_min_vals, label='t1')
plt.plot(phase_arr_th, t2_min_vals, label='t2')
plt.title('Minimizing values.')
plt.legend()
plt.show()
