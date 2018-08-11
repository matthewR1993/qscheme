import numpy as np
import matplotlib.pyplot as plt


det = 'FIRST'

phases = [0.0, 0.125, 0.25, 0.375]
size = len(phases)

dX_min_arr = np.zeros(size, dtype=complex)
dP_min_arr = np.zeros(size, dtype=complex)
epr_x_min_arr = np.zeros(size, dtype=complex)
epr_p_min_arr = np.zeros(size, dtype=complex)
uncert_min_arr = np.zeros(size, dtype=complex)

for i in range(size):
    phase = phases[i]

    # save_root = '/Users/matvei/PycharmProjects/qscheme/results/res14/'
    save_root = '/home/matthew/qscheme/results/res14/'
    fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(phase, det)

    fl = np.load(save_root + fname)

    sqeez_dX = fl[2]
    sqeez_dP = fl[3]
    erp_correl_x = fl[4]
    erp_correl_p = fl[5]

    uncert_min_arr[i] = np.amin(np.multiply(sqeez_dX, sqeez_dP))
    dX_min_arr[i] = np.amin(sqeez_dX)
    dP_min_arr[i] = np.amin(sqeez_dP)
    epr_x_min_arr[i] = np.amin(erp_correl_x)
    epr_p_min_arr[i] = np.amin(erp_correl_p)


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

