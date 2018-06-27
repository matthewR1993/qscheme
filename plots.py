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

