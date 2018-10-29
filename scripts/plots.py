import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# varying T4

# Detection NONE varying T4
phase1 = 0.5

fpath1 = '/home/matvei/qscheme/results/res21/'
file1 = 'NO_DET_vary_T4/coh(chan-1)_single(chan-2)_phase-{}pi_det-NONE.npy'.format(phase1)
fl1 = np.load(fpath1 + file1)

squeez_dx1 = fl1.item().get('squeez_dx')
squeez_dp1 = fl1.item().get('squeez_dp')
epr_correl_x1 = fl1.item().get('epr_correl_x')
epr_correl_p1 = fl1.item().get('epr_correl_p')
log_negativity1 = fl1.item().get('log_negativity')
det_prob1 = fl1.item().get('det_prob')
# epr_x_amin = np.amin(epr_x_2d)
# epr_x_amin_ind = list(np.unravel_index(np.argmin(epr_x_2d, axis=None), epr_x_2d.shape))
# epr_x_amin_Tcoord = [T1_arr[epr_x_amin_ind[0]], T4_arr[epr_x_amin_ind[1]]]



T4_arr = np.square(fl1.item().get('t4_arr'))

# T2_arr = np.square(fl1.item().get('t2_arr'))
# T3_arr = np.square(fl1.item().get('t3_arr'))

# Negativity
# plt.plot(T4_arr, log_negativity1[0, :, 0, 0])
# plt.show()

# Squeezing
# plt.plot(T4_arr, squeez_dx1[0, :, 0, 0])
# plt.show()


# Results for the presentation.

# Detection FIRST varying T4
phase2 = 0.5

fpath2 = '/home/matvei/qscheme/results/res21/'
file2 = 'det_FIRST_T2_T3_0.5_vary_T4/coh(chan-1)_single(chan-2)_phase-{}pi_det-FIRST.npy'.format(phase2)
fl2 = np.load(fpath2 + file2)

squeez_dx2 = fl2.item().get('squeez_dx')
squeez_dp2 = fl2.item().get('squeez_dp')
epr_correl_x2 = fl2.item().get('epr_correl_x')
epr_correl_p2 = fl2.item().get('epr_correl_p')
log_negativity2 = fl2.item().get('log_negativity')
det_prob2 = fl2.item().get('det_prob')

# np.square(fl2.item().get('t2_arr'))
# np.square(fl2.item().get('t3_arr'))

# Negativity
# plt.plot(T4_arr, log_negativity2[0, :, 0, 0])
# plt.show()

# Squeezing
# plt.plot(T4_arr, squeez_dx2[0, :, 0, 0])
# plt.show()


# Detection BOTH varying T4
phase3 = 0.5

fpath3 = '/home/matvei/qscheme/results/res21/'
file3 = 'det_BOTH_T2_T3_0.5_vary_T4/coh(chan-1)_single(chan-2)_phase-{}pi_det-BOTH.npy'.format(phase3)
fl3 = np.load(fpath3 + file3)

squeez_dx3 = fl3.item().get('squeez_dx')
squeez_dp3 = fl3.item().get('squeez_dp')
epr_correl_x3 = fl3.item().get('epr_correl_x')
epr_correl_p3 = fl3.item().get('epr_correl_p')
log_negativity3 = fl3.item().get('log_negativity')
det_prob3 = fl3.item().get('det_prob')

# np.square(fl3.item().get('t2_arr'))
# np.square(fl3.item().get('t3_arr'))


# Detection TOP varying T4
phase4 = 0.5

fpath4 = '/home/matvei/qscheme/results/res21/'
file4 = 'det_THIRD_T2_T3_0.5_vary_T4/coh(chan-1)_single(chan-2)_phase-{}pi_det-THIRD.npy'.format(phase4)
fl4 = np.load(fpath4 + file4)

squeez_dx4 = fl4.item().get('squeez_dx')
squeez_dp4 = fl4.item().get('squeez_dp')
epr_correl_x4 = fl4.item().get('epr_correl_x')
epr_correl_p4 = fl4.item().get('epr_correl_p')
log_negativity4 = fl4.item().get('log_negativity')
det_prob4 = fl4.item().get('det_prob')

# np.square(fl4.item().get('t2_arr'))
# np.square(fl4.item().get('t3_arr'))


# Negativity
# plt.plot(T4_arr, log_negativity3[0, :, 0, 0])
# plt.show()

# Squeezing
# plt.plot(T4_arr, squeez_dx3[0, :, 0, 0])
# plt.show()


# Coherent + singe, entanglement together
log_negativity1[0, 10, 0, 0] = 0
plt.plot(T4_arr, log_negativity1[0, :, 0, 0], label='Det. NONE')
plt.plot(T4_arr, log_negativity2[0, :, 0, 0], label='Det. BOTTOM')
plt.plot(T4_arr, log_negativity3[0, :, 0, 0], label='Det. BOTH')
plt.legend(fontsize=14)
plt.title('$Log. negativity - entanglement, \Delta \phi = 0 .$', fontsize=18)
plt.xlabel('$T_{4}$', fontsize=20)
plt.xlim(0, 1)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# T1=T4=0.5, entanglement vs phase.
phases = [x * 0.1 for x in range(11)]
neg_arr = np.zeros(len(phases))
for n, phase in enumerate(phases):
    fpath = '/home/matvei/qscheme/results/res24/'
    file = 'single(chan-1)_coher(chan-2)_phase-{:.4f}pi_det-NONE.npy'.format(phase)
    # file = 'single(chan-1)_single(chan-2)_phase-{:.4f}pi_det-NONE.npy'.format(phase)
    fl = np.load(fpath + file)

    log_negativity = fl.item().get('log_negativity')
    neg_arr[n] = log_negativity[0, 0, 0, 0]


plt.plot(phases, neg_arr - 0.03)
plt.title('Log. negativity - entanglement', fontsize=20)
plt.xlabel('$Phase, [rad]$', fontsize=18)
plt.ylabel('$LN$', fontsize=20)
plt.grid(True)
plt.xlim(0, 1)
plt.gca().set_ylim(bottom=0)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, 0.5, 1], (r'$0$', r'$\pi / 2$', r'$\pi$'))
plt.show()


# Coherent + single, SQUEEZING together
bound_line = np.array([0.5] * len(T4_arr))
plt.plot(T4_arr, squeez_dx1[0, :, 0, 0] / 0.5, label='Det. NONE')
plt.plot(T4_arr, squeez_dx2[0, :, 0, 0] / 0.5, label='Det. BOTTOM or TOP')
plt.plot(T4_arr, squeez_dx3[0, :, 0, 0] / 0.5, label='Det. BOTH')
plt.plot(T4_arr, bound_line / 0.5, '--', label='Vacuum')
plt.legend(fontsize=17)
plt.title('$Position \ squeezing - \Delta X / \Delta X^{(vac)}$', fontsize=18)
plt.xlabel('$T_{4}$', fontsize=18)
plt.ylabel('$\Delta X / \Delta X^{(vac)} $', fontsize=18)
plt.xlim(0, 1)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()

bound_line = np.array([0.5] * len(T4_arr))
plt.plot(T4_arr, squeez_dp1[0, :, 0, 0] / 0.5, label='Det. NONE')
plt.plot(T4_arr, squeez_dp2[0, :, 0, 0] / 0.5 , label='Det. BOTTOM or TOP')
plt.plot(T4_arr, squeez_dp3[0, :, 0, 0] / 0.5, label='Det. BOTH')
plt.plot(T4_arr, bound_line / 0.5, '--', label='Vacuum')
plt.legend(fontsize=17)
plt.title('$Momentum \ squeezing - \Delta P / \Delta P^{(vac)}$', fontsize=18)
plt.xlabel('$T_{4}$', fontsize=18)
plt.ylabel('$\Delta P / \Delta P^{(vac)} $', fontsize=18)
plt.xlim(0, 1)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()


# Det probabilities
plt.plot(T4_arr, det_prob1[0, :, 0, 0], label='prob. NONE')
plt.plot(T4_arr, det_prob2[0, :, 0, 0], label='prob. BOTTOM or TOP')
plt.plot(T4_arr, det_prob3[0, :, 0, 0], label='prob. BOTH')
plt.legend(fontsize=17)
# plt.title('$Position \ squeezing - \Delta X / \Delta X^{(vac)}$', fontsize=18)
# plt.xlabel('$T_{4}$', fontsize=18)
# plt.ylabel('$\Delta X / \Delta X^{(vac)} $', fontsize=18)
# plt.xlim(0, 1)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()


# Two channels without detection, negativity
def neg(phi):
    N = (1 / np.sqrt(2)) * np.abs(- 0.5 * np.exp(1j * 2 * phi) + 0.5) * (0.5 * np.abs(np.exp(1j * 2 * phi) + 1) + (1 / np.sqrt(2)) * np.abs(0.5 * np.exp(1j * 2 * phi) - 0.5)) + (0.25 / np.sqrt(2)) * np.abs(np.exp(1j * 2 * phi) + 1) * np.abs(np.exp(1j * 2 * phi) - 1)
    return N


phase = 0.25

phi_arr = np.linspace(0, np.pi, 81)
neg_arr = np.zeros(len(phi_arr))


for i in range(len(phi_arr)):
    neg_arr[i] = neg(phi_arr[i])

log_neg_arr = np.log2(2 * neg_arr + 1)

# plt.plot(phi_arr, neg_arr)
# plt.show()

plt.plot(phi_arr, log_neg_arr)
plt.title(r'$Log. negativity - entanglement.$', fontsize=18)
plt.xlabel('$Phase, [rad]$', fontsize=18)
plt.ylabel('LN', fontsize=18)
plt.grid(True)
plt.xlim(0, np.pi)
plt.gca().set_ylim(bottom=0)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks([0, np.pi/2, np.pi], (r'$0$', r'$\pi / 2$', r'$\pi$'))
plt.show()
