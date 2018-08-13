import numpy as np
import matplotlib.pyplot as plt


det = 'FIRST'

phases = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
size = len(phases)

dX_min_arr = np.zeros(size, dtype=complex)
dP_min_arr = np.zeros(size, dtype=complex)
epr_x_min_arr = np.zeros(size, dtype=complex)
epr_p_min_arr = np.zeros(size, dtype=complex)
uncert_min_arr = np.zeros(size, dtype=complex)

dX_min_ind = np.zeros(size, dtype=list)
dP_min_ind = np.zeros(size, dtype=list)
epr_x_min_ind = np.zeros(size, dtype=list)
epr_p_min_ind = np.zeros(size, dtype=list)
uncert_min_ind = np.zeros(size, dtype=list)


for i in range(size):
    phase = phases[i]

    save_root = '/Users/matvei/PycharmProjects/qscheme/results/res14/'
    # save_root = '/home/matthew/qscheme/results/res14/'
    fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(phase, det)

    fl = np.load(save_root + fname)

    sqeez_dX = fl.item().get('squeez_dx')
    sqeez_dP = fl.item().get('squeez_dp')
    erp_correl_x = fl.item().get('epr_correl_x')
    erp_correl_p = fl.item().get('epr_correl_p')

    uncert_min_arr[i] = np.amin(np.multiply(sqeez_dX, sqeez_dP))
    dX_min_arr[i] = np.amin(sqeez_dX)
    dP_min_arr[i] = np.amin(sqeez_dP)
    epr_x_min_arr[i] = np.amin(erp_correl_x)
    epr_p_min_arr[i] = np.amin(erp_correl_p)

    uncert = np.multiply(sqeez_dX, sqeez_dP)

    uncert_min_ind[i] = list(np.unravel_index(np.argmax(uncert, axis=None), uncert.shape))
    dX_min_ind[i] = list(np.unravel_index(np.argmax(sqeez_dX, axis=None), sqeez_dX.shape))
    dP_min_ind[i] = list(np.unravel_index(np.argmax(sqeez_dP, axis=None), sqeez_dP.shape))
    epr_x_min_ind[i] = list(np.unravel_index(np.argmax(erp_correl_x, axis=None), erp_correl_x.shape))
    epr_p_min_ind[i] = list(np.unravel_index(np.argmax(erp_correl_p, axis=None), erp_correl_p.shape))


# TODO min for which parameters.?

plt.plot(phases, uncert_min_arr)
plt.title('dPdX')
plt.show()

plt.plot(phases, dX_min_arr)
plt.title('dX min')
plt.show()

plt.plot(phases, dP_min_arr)
plt.title('dP min')
plt.show()

plt.plot(phases, epr_x_min_arr)
plt.title('EPR X_min')
plt.show()

plt.plot(phases, epr_p_min_arr)
plt.title('EPR P_min')
plt.show()

