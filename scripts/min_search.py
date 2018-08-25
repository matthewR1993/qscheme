import numpy as np
import matplotlib.pyplot as plt


det = 'FIRST'

# phases = [x * 0.125 for x in range(9)]
phases = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
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


# Uncertainty.
plt.plot(phases, uncert_min_arr, 'r.')
plt.title('$dPdX^{min}$')
plt.ylabel('$dPdX^min}$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

# Quadratures.
plt.plot(phases, dX_min_arr, 'r.')
plt.title('$dX^{min}$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

plt.plot(phases, dP_min_arr, 'r.')
plt.title('$dP^{min}$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

# EPR:
plt.plot(phases, epr_x_min_arr, 'r.')
plt.title('$EPR \ X^{min}$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

plt.plot(phases, epr_p_min_arr, 'r.')
plt.title('$EPR \ P^{min}$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

# Probabilities of realisation for EPR:
plt.plot(phases, epr_x_min_prob_arr, 'r.')
plt.title('$P[EPR \ X]$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()

plt.plot(phases, epr_p_min_prob_arr, 'r.')
plt.title('$P[EPR \ P]$')
plt.xlabel('$phase \ in \ \pi$')
plt.show()


