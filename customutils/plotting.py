# Script for figures generation using saved files.

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res13/coh(ch1)_single(ch2)_var_phase_t1_t4_det-FIRST/phase-0.5pi/'
# save_root = '/home/matthew/qscheme/results/res13/coh(ch2)_single(ch1)_var_phase_t1_t4_det-FIRST/phase-0.0pi/'
fname = 'phase_0.5pi.npy'

# fle = np.load(save_root + fname)
# log_negativity = fle[0]
# mut_information = fle[1]
# sqeez_dX = fle[2]
# sqeez_dP = fle[3]
# erp_correl_x = fle[4]
# erp_correl_p = fle[5]


T4_min, T4_max = 0, 1
T1_min, T1_max = 0, 1
ORTS = [T4_min, T4_max, T1_min, T1_max]


# # 2D pictures
# # Varying t4
# plt.plot(np.square(t4_array), np.real(log_entropy_subs1_array[:, 0]), label=r'$Log. FN \ entropy \ subs \ 1$')
# plt.plot(np.square(t4_array), np.real(log_entropy_subs2_array[:, 0]), label=r'$Log. FN \ entropy \ subs \ 2$')
# plt.plot(np.square(t4_array), np.real(mut_information[:, 0]), label=r'$Mut \ information, \ FN \ log. \ entropy.$')
# plt.plot(np.square(t4_array), np.real(full_fn_entropy[:, 0]), label=r'$Full \ FN \ log. \ entropy.$')
# # plt.plot(np.square(t4_array), np.real(lin_entropy[:, 0]), label=r'$Lin. entropy$')
# plt.plot(np.square(t4_array), np.real(log_negativity[:, 0]), label=r'$Log. \ negativity$')
# # plt.plot(np.square(t4_array), np.real(log_negativity_aftdet[:, 0]), label=r'$Log. negativity, after det. with phase$')
# # plt.title(f'Entanglement. Phase=%0.2f pi. Det. - %s' % (ph_inpi, DET_CONF))
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel('$Entanglement$')
# # plt.title('Phase = {0}pi'.format(ph_inpi))
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # plot squeezing
# plt.plot(np.square(t4_array), 10*np.log10(sqeez_dX[:, 0]/QUADR_VAR_X_VAC), label=r'$10\log_{10}{\frac{\Delta X^{(out)}}{\Delta X^{(vac)}}}$')
# plt.plot(np.square(t4_array), 10*np.log10(sqeez_dP[:, 0]/QUADR_VAR_P_VAC), label=r'$10\log_{10}{\frac{\Delta P^{(out)}}{\Delta P^{(vac)}}}$')
# # plt.plot(np.square(t4_array), np.multiply(sqeez_dX[:, 0], sqeez_dP[:, 0]), label=r'$\Delta X \Delta P$')
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel('$Squeezing$')
# plt.title('$Squeezing$')
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # ERP correlations
# plt.plot(np.square(t4_array), erp_correl_x[:, 0]/EPR_VAR_X_VAC, label=r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$')
# plt.plot(np.square(t4_array), erp_correl_p[:, 0]/EPR_VAR_P_VAC, label=r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$')
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel('$ERP \ correlations \ [a. u.]$')
# plt.title('$ERP \ correlations$')
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # dX
# # dX - 3D - picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# # surf = ax.plot_surface(X, Y, np.real(sqeez_dX), cmap=cm.coolwarm,
# #                         linewidth=0, antialiased=False)
# # plt.title(r'$\Delta X$')
# surf = ax.plot_surface(X, Y, 10*np.log10(np.real(sqeez_dX)/QUADR_VAR_X_VAC), cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$10\log_{10}{(\frac{\Delta X^{(out)}}{\Delta X^{(vac)}})}$', fontsize=16)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# dX - 2D - picture.
# im = plt.imshow(np.real(sqeez_dX), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower', extent=ORTS)
# plt.title('$\Delta X$')
# im = plt.imshow(10*np.log10(np.real(sqeez_dX[:, :, 0, 0])/QUADR_VAR_X_VAC), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$10\log_{10}{(\frac{\Delta X^{(out)}}{\Delta X^{(vac)}})}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{1}$', fontsize=16)
# plt.ylabel(r'$T_{4}$', fontsize=16)
# plt.show()
#
#
# # dP - 3D - picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# # surf = ax.plot_surface(X, Y, np.real(sqeez_dP), cmap=cm.coolwarm,
# #                        linewidth=0, antialiased=False)
# # plt.title(r'$\Delta P$')
# surf = ax.plot_surface(X, Y, 10*np.log10(np.real(sqeez_dP)/QUADR_VAR_P_VAC), cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$10*\log_{10}{(\frac{\Delta P^{(out)}}{\Delta P^{(vac)}})}$')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# dP - 2D - picture.
# im = plt.imshow(np.real(sqeez_dP), interpolation='None', cmap=cm.RdYlGn, origin='lower', extent=ORTS)
# plt.title('$\Delta P$')
# im = plt.imshow(10*np.log10(np.real(sqeez_dP[:, :, 0, 0])/QUADR_VAR_P_VAC), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$10\log_{10}{(\frac{\Delta P^{(out)}}{\Delta P^{(vac)}})}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# Uncertainty principle: dX * dP >= 1/4.
# Should be >= 1/4
# im = plt.imshow(np.multiply(np.real(sqeez_dX[:, :, 0, 0]), np.real(sqeez_dP[:, :, 0, 0])), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$\Delta X\Delta P$')
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# # EPR correlations:
# # EPR X, 3D picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# surf = ax.plot_surface(X, Y, np.real(erp_correl_x)/EPR_VAR_X_VAC, cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.xlim(1, 0)
# plt.ylim(0, 1)
# plt.show()
#
#
# # ERP X, 2D picture
# im = plt.imshow(np.real(erp_correl_x[:, :, 0, 0])/EPR_VAR_X_VAC, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()

# Trimmed.
# size = len(erp_correl_x)
# Z_erpX = np.real(erp_correl_x[:, :, 0, 0])/EPR_VAR_X_VAC
# for i in range(size):
#     for j in range(size):
#         if Z_erpX[i, j] > 1:
#             Z_erpX[i, j] = 0
# im = plt.imshow(Z_erpX, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$\frac{\Delta[X^{(1)} - X^{(2)}]^{(out)}}{\Delta[X^{(1)} - X^{(2)}]^{(vac)}}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
#plt.show()


# # ERP P, 3D picture
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
# surf = ax.plot_surface(X, Y, np.real(erp_correl_p)/EPR_VAR_P_VAC, cmap=cm.Spectral,
#                        linewidth=0, antialiased=False)
# plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.xlim(1, 0)
# plt.ylim(0, 1)
# plt.show()

# ERP P, 2D picture
# im = plt.imshow(np.real(erp_correl_p[:, :, 0, 0])/EPR_VAR_P_VAC, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()

# Trimmed.
# size = len(erp_correl_p)
# Z_erpP = np.real(erp_correl_p[:, :, 0, 0])/EPR_VAR_P_VAC
# for i in range(size):
#     for j in range(size):
#        if Z_erpP[i, j] > 1:
#            Z_erpP[i, j] = 0
# im = plt.imshow(Z_erpP, interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$\frac{\Delta[P^{(1)} + P^{(2)}]^{(out)}}{\Delta[P^{(1)} + P^{(2)}]^{(vac)}}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()


# Uncertainty principle for EPR, should be > 1/2???
# im = plt.imshow(np.multiply(np.real(erp_correl_x[:, :, 0, 0]), np.real(erp_correl_p[:, :, 0, 0])), interpolation='None', cmap=cm.Spectral, origin='lower', extent=ORTS)
# plt.title(r'$\Delta[X^{(1)} - X^{(2)}]^{(out)}\Delta[P^{(1)} + P^{(2)}]^{(out)}$', fontsize=16)
# plt.colorbar(im)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
