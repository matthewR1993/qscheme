import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


filepath = '/home/matthew/qscheme/results/res18/coh(chan-1)_single(chan-2)_phase-0.0pi_det-FIRST_T1_T4.npy'
fl = np.load(filepath)


T1_arr = np.square(fl.item().get('t1_arr'))
T4_arr = np.square(fl.item().get('t4_arr'))
epr_x = fl.item().get('epr_correl_x')
epr_x_2d = np.real(epr_x[:, :, 0, 0])

epr_x_amin = np.amin(epr_x_2d)
epr_x_amin_ind = list(np.unravel_index(np.argmin(epr_x_2d, axis=None), epr_x_2d.shape))
epr_x_amin_Tcoord = [T1_arr[epr_x_amin_ind[0]], T4_arr[epr_x_amin_ind[1]]]


# epr x plot
plt.imshow(epr_x_2d, origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.scatter(x=[epr_x_amin_ind[1]], y=[epr_x_amin_ind[0]], c='r', s=60, marker='+')
plt.xlabel('T4')
plt.ylabel('T1')
plt.show()


# Uncertanity
squeez_dx = fl.item().get('squeez_dx')
squeez_dp = fl.item().get('squeez_dp')
dxdy = np.real(np.multiply(squeez_dx[:, :, 0, 0], squeez_dp[:, :, 0, 0]))

# works
plt.imshow(dxdy)
plt.colorbar()
plt.show()


# over T2, T3
filepath2 = '/home/matthew/qscheme/results/res18/coh(chan-1)_single(chan-2)_phase-0.0pi_det-FIRST_T2_T3.npy'
fl2 = np.load(filepath2)


T2_arr = np.square(fl2.item().get('t2_arr'))
T3_arr = np.square(fl2.item().get('t3_arr'))
epr_x = fl2.item().get('epr_correl_x')
epr_x_2d = np.real(epr_x[0, 0, :, :])

epr_x_amin = np.amin(epr_x_2d)
epr_x_amin_ind = list(np.unravel_index(np.argmin(epr_x_2d, axis=None), epr_x_2d.shape))
epr_x_amin_Tcoord = [T2_arr[epr_x_amin_ind[0]], T3_arr[epr_x_amin_ind[1]]]


# epr x plot
plt.imshow(epr_x_2d, origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.scatter(x=[epr_x_amin_ind[1]], y=[epr_x_amin_ind[0]], c='r', s=60, marker='+')
plt.xlabel('T2')
plt.ylabel('T3')
plt.show()


# Uncertanity
squeez_dx = fl2.item().get('squeez_dx')
squeez_dp = fl2.item().get('squeez_dp')
dxdy = np.real(np.multiply(squeez_dx[0, 0, :, :], squeez_dp[0, 0, :, :]))

# works
plt.imshow(dxdy)
plt.colorbar()
plt.show()

#
phase = 0.25

fpath = '/Users/matvei/PycharmProjects/qscheme/results/res21/'
file = 'NO_DET_vary_T4/coh(chan-1)_single(chan-2)_phase-{}pi_det-NONE.npy'.format(phase)
fl = np.load(fpath + file)

squeez_dx = fl.item().get('squeez_dx')
squeez_dp = fl.item().get('squeez_dp')
epr_correl_x = fl.item().get('epr_correl_x')
epr_correl_p = fl.item().get('epr_correl_p')
log_negativity = fl.item().get('log_negativity')


T4_arr = np.square(fl.item().get('t4_arr'))

# plt.plot(T4_arr, log_negativity[0, :, 0, 0])
# plt.show()

plt.plot(T4_arr, epr_correl_x[0, :, 0, 0])
plt.show()
