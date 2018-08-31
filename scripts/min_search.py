import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt


QUADR_VAR_X_VAC = 1/2
QUADR_VAR_P_VAC = 1/2

# det = 'FIRST'
det = 'THIRD'

# phases = [x * 0.25 for x in range(9)]
phases = [x * 0.125 for x in range(17)]
size = len(phases)

dX_min_arr = np.zeros(size, dtype=complex)
dP_min_arr = np.zeros(size, dtype=complex)
epr_x_min_arr = np.zeros(size, dtype=complex)
epr_p_min_arr = np.zeros(size, dtype=complex)
uncert_min_arr = np.zeros(size, dtype=complex)
epr_x_min_prob_arr = np.zeros(size, dtype=complex)
epr_p_min_prob_arr = np.zeros(size, dtype=complex)

dX_min_ind = np.zeros(size, dtype=list)
dP_min_ind = np.zeros(size, dtype=list)
epr_x_min_ind = np.zeros(size, dtype=list)
epr_p_min_ind = np.zeros(size, dtype=list)
uncert_min_ind = np.zeros(size, dtype=list)


for i in range(size):
    phase = phases[i]

    save_root = '/Users/matvei/PycharmProjects/qscheme/results/res15/'
    # save_root = '/home/matthew/qscheme/results/res14/'
    fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(phase, det)

    fl = np.load(save_root + fname)

    sqeez_dX = fl.item().get('squeez_dx')
    sqeez_dP = fl.item().get('squeez_dp')
    erp_correl_x = fl.item().get('epr_correl_x')
    erp_correl_p = fl.item().get('epr_correl_p')
    prob = fl.item().get('det_prob')

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


line = [1] * len(phases)

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
plt.title(r'$\frac{1}{\sqrt{2}} \ \Delta[X^{(1)} - X^{(2)}]^{(out)}$')
plt.plot(phases, line, '-.')
plt.xlabel('$Phase, [\pi]$')
plt.grid(True)
plt.show()

plt.plot(phases, epr_p_min_arr / sqrt(1/2), 'b-o')
plt.title(r'$\frac{1}{\sqrt{2}} \ \Delta[P^{(1)} + P^{(2)}]^{(out)}$')
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

# Minimizing T points.
dX_min_ind_arr = np.array([np.array(x) for x in dX_min_ind]) / 10
dP_min_ind_arr = np.array([np.array(x) for x in dP_min_ind]) / 10
epr_x_min_ind_arr = np.array([np.array(x) for x in epr_x_min_ind]) / 10
epr_p_min_ind_arr = np.array([np.array(x) for x in epr_p_min_ind]) / 10


df_phase = pd.DataFrame(np.array(phases))

df_dX_min_ind = pd.concat([df_phase, pd.DataFrame(dX_min_ind_arr)], axis=1, ignore_index=True)
df_dX_min_ind.columns = ['Phase, [pi]', 'T1', 'T4', 'T2', 'T3']
df_dP_min_ind = pd.concat([df_phase, pd.DataFrame(dP_min_ind_arr)], axis=1, ignore_index=True)
df_dP_min_ind.columns = ['Phase, [pi]', 'T1', 'T4', 'T2', 'T3']
df_epr_x_min_ind = pd.concat([df_phase, pd.DataFrame(epr_x_min_ind_arr)], axis=1, ignore_index=True)
df_epr_x_min_ind.columns = ['Phase, [pi]', 'T1', 'T4', 'T2', 'T3']
df_epr_p_min_ind = pd.concat([df_phase, pd.DataFrame(epr_p_min_ind_arr)], axis=1, ignore_index=True)
df_epr_p_min_ind.columns = ['Phase, [pi]', 'T1', 'T4', 'T2', 'T3']


# Plot tables.
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
df = df_dX_min_ind
# df = df_dP_min_ind
# df = df_epr_x_min_ind
# df = df_epr_p_min_ind
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
fig.tight_layout()
plt.show()
