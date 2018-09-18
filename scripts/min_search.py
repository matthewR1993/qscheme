import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt


QUADR_VAR_X_VAC = 1/2
QUADR_VAR_P_VAC = 1/2

r1_grid = 11
r4_grid = 11
r2_grid = 11
r3_grid = 11

# det = 'FIRST'
det = 'THIRD'
# det = 'NONE'
# det = 'BOTH'

quant = 'EPR_X'

# phases = [x * 0.25 for x in range(9)]
phases = [x * 0.125 for x in range(17)]
# phases = [0.25]

size = len(phases)

line = [1] * len(phases)

crit_prob = 0.1

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


for i in range(size):
    print('step:', i)
    phase = phases[i]

    # save_root = '/home/matthew/qscheme/results/res19_rough/'
    # save_root = '/Users/matvei/PycharmProjects/qscheme/results/res19_rough/'
    # fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(phase, det)

    # save_root = '/home/matthew/qscheme/results/res19_incr_accuracy/'
    save_root = '/Users/matvei/PycharmProjects/qscheme/results/res19_incr_accuracy/'
    fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}_quant-{}.npy'.format(phase, det, quant)

    fl = np.load(save_root + fname)

    sqeez_dX = fl.item().get('squeez_dx')
    sqeez_dP = fl.item().get('squeez_dp')
    erp_correl_x = fl.item().get('epr_correl_x')
    erp_correl_p = fl.item().get('epr_correl_p')
    prob = fl.item().get('det_prob')

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


# Uncertainty.
plt.plot(phases, uncert_min_arr, 'r.')
plt.title('$dPdX^{min}$')
plt.ylabel('$dPdX^min}$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

# Quadratures.
plt.plot(phases, 10*np.log10(dX_min_arr/QUADR_VAR_X_VAC), 'r-o')
plt.title(r'$10\log_{10}{\frac{\Delta X^{(out)}}{\Delta X^{(vac)}}}$')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.show()

plt.plot(phases, 10*np.log10(dP_min_arr/QUADR_VAR_P_VAC), 'b-o')
plt.title(r'$10\log_{10}{\frac{\Delta P^{(out)}}{\Delta P^{(vac)}}}$')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.show()

# EPR:
plt.plot(phases, epr_x_min_arr / sqrt(1/2), 'r-o')
plt.title(r'$\sqrt{2} \ \Delta[X^{(1)} - X^{(2)}]^{(out)}$')
plt.plot(phases, line, '-.')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.show()

plt.plot(phases, epr_p_min_arr / sqrt(1/2), 'b-o')
plt.title(r'$\sqrt{2} \ \Delta[P^{(1)} + P^{(2)}]^{(out)}$')
plt.plot(phases, line, '-.')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
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

t1_arr = fl.item().get('t1_arr')
t4_arr = fl.item().get('t4_arr')
t2_arr = fl.item().get('t2_arr')
t3_arr = fl.item().get('t3_arr')

T1_arr = np.square(t1_arr)
T4_arr = np.square(t4_arr)
T2_arr = np.square(t2_arr)
T3_arr = np.square(t3_arr)

# Minimizing 'T' points.
dX_min_T_arr = np.array([np.array([T1_arr[x[0]], T4_arr[x[1]], T2_arr[x[2]], T3_arr[x[3]]]) for x in dX_min_ind])
dP_min_T_arr = np.array([np.array([T1_arr[x[0]], T4_arr[x[1]], T2_arr[x[2]], T3_arr[x[3]]]) for x in dP_min_ind])
epr_x_min_T_arr = np.array([np.array([T1_arr[x[0]], T4_arr[x[1]], T2_arr[x[2]], T3_arr[x[3]]]) for x in epr_x_min_ind])
epr_p_min_T_arr = np.array([np.array([T1_arr[x[0]], T4_arr[x[1]], T2_arr[x[2]], T3_arr[x[3]]]) for x in epr_p_min_ind])


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


import matplotlib.cm as cm

t1 = fl.item().get("t1_arr")
t4 = fl.item().get("t4_arr")


plt.imshow(np.real(prob[4, 0, :, :]), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
#plt.scatter(x=[epr_x_amin_ind[1]], y=[epr_x_amin_ind[0]], c='r', s=80, marker='+')
#plt.scatter(x=[50], y=[50], c='g', s=80, marker='+')
#plt.plot(T1_coord*100, T4_coord*100)
plt.xlabel('T3')
plt.ylabel('T2')
plt.show()
