# Dens matrix nad mean photon number
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA


wlen = 5

a = 0.1  # 1 and 0.1
b = 1

ampl = lambda wi, ws: np.exp(-(wlen - ws - wi)**2/(2*a**2)) * np.exp(-(ws - wi)**2/(2*b**2))

grd = 140

wi_arr = np.linspace(-6, 6, grd)
ws_arr = np.linspace(-6, 6, grd)

ampl_arr = np.zeros((grd, grd))

for i in range(grd):
    for j in range(grd):
        ampl_arr[i, j] = ampl(wi_arr[i], wi_arr[j])


# Plot 3d
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# X = wi_arr
# Y = ws_arr
# X, Y = np.meshgrid(X, Y)
#
# surf = ax.plot_surface(X, Y, ampl_arr, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$w_i$', fontsize=16)
# plt.ylabel(r'$w_s$', fontsize=16)
# plt.show()
#
#
# # Plot 2d
# im = plt.imshow(ampl_arr, interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')
# plt.xlabel(r'$w_i$', fontsize=16)
# plt.ylabel(r'$w_s$', fontsize=16)
# plt.show()


# shmidt number
dens_matrix = np.zeros((grd, grd))


for i in range(grd):
    for j in range(grd):
        mult_arr = np.zeros(grd)
        for k in range(grd):
            mult_arr[k] = ampl(wi_arr[i], ws_arr[k]) * ampl(wi_arr[j], ws_arr[k])
        dens_matrix[i, j] = np.trapz(mult_arr, x=wi_arr)


# w - values
w, v = LA.eig(dens_matrix)

w_norm = w / np.sum(w)

# plot eigenvalues
plt.plot(list(range(25)), w_norm[0:25], '-o')
plt.show()

# shmidt number  K = 5
shm_num = 1 / np.sum(np.square(w_norm))

# mean photon number vs signal
mean_num = np.zeros(grd)

for i in range(grd):
    int_arr = np.zeros(grd)
    for m in range(grd):
        int_arr[m] = np.abs(ampl(wi_arr[m], ws_arr[i]))
    mean_num[i] = np.trapz(int_arr, x=wi_arr)


plt.plot(ws_arr, mean_num)
plt.xlabel(r'$w_s$', fontsize=16)
plt.ylabel(r'$N_s$', fontsize=16)
plt.show()


# prob distribution
tau_grd = 81
tau_arr = np.linspace(-10, 10, tau_grd)

coincid_prob_arr = np.zeros(tau_grd)

for n in range(tau_grd):
    int_arr = np.zeros((grd, grd))
    for i in range(grd):
        for j in range(grd):
            int_arr[i, j] = ampl(wi_arr[i], ws_arr[j])*ampl(wi_arr[j], ws_arr[i])*(1 - np.cos((wi_arr[i] - ws_arr[j])*tau_arr[n]))
    int1 = np.trapz(int_arr, x=wi_arr)
    coincid_prob_arr[n] = np.trapz(int1, x=wi_arr)


plt.plot(tau_arr, coincid_prob_arr)
plt.xlabel(r'$\tau$', fontsize=16)
plt.ylabel(r'$P$', fontsize=16)
plt.show()


# mult = 1e-3
#
# plt.plot(np.square(t4_array), np.real(log_entropy_array), label=r'$Log. entropy$')
# plt.plot(np.square(t4_array), np.real(lin_entropy), label=r'$Lin. entropy$')
# plt.plot(np.square(t4_array), np.real(mult*log_negativity), label=r'$Log. negativity*10^{-3}$')
# plt.title(r'Entanglement.')
# plt.xlabel(r'$t_{4}$', fontsize=16)
# plt.xlim([0, 1])
# plt.legend()
# plt.grid(True)
# plt.show()


# Entropy S(t1, t4) 3D plot.
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# X = np.square(t4_array)
# Y = np.square(t1_array)
# X, Y = np.meshgrid(X, Y)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, np.real(log_entropy_array), cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
# plt.title(r'Log. VN entropy.')
#
# # surf = ax.plot_surface(X, Y, np.real(log_negativity), cmap=cm.coolwarm,
# #                        linewidth=0, antialiased=False)
# # plt.title(r'Log. negativity.')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel(r'$T_{4}$', fontsize=16)
# plt.ylabel(r'$T_{1}$', fontsize=16)
# plt.show()
#
#
# # Entropy S(t1, t4) 2D plot.
# im = plt.imshow(np.real(log_entropy_array), cmap=cm.RdBu)  # Log. entropy
# # im = plt.imshow(np.real(log_negativity), cmap=cm.RdBu)  # Log. nagativity
# cset = plt.contour(np.real(log_entropy_array), np.arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
# plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
# plt.colorbar(im)
# plt.xlabel(r'$t_{4}$', fontsize=16)
# plt.ylabel(r'$t_{1}$', fontsize=16)
# # plt.title('$S(t_{4}, t_{1}) - VN \ entropy$')
# plt.title('$S(t_{4}, t_{1}) - Log. \ negativity$')
# plt.show()


'''

# plot input states
plt.bar(list(range(len(input_st))), input_st, width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], width=1, edgecolor='c')
plt.title('Input state')
plt.xlabel('Number of photons')
plt.show()
plt.bar(list(range(len(auxiliar y_st))), auxiliary_st, color='g', width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], color='g', width=1, edgecolor='c')
plt.title('Auxiliary state')
plt.xlabel('Number of photons')
plt.show()

'''
