import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qutip import (wigner, super_tensor, Qobj)
from time import gmtime, strftime

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *


def dens_matrix(phase):
    # transposed reorganised matrix
    # all indexes are -1
    rho = np.zeros((9, 9), dtype=complex)
    rho[2, 2] = (1/8) * np.abs(1 - np.exp(1j*2*phase))**2
    rho[4, 4] = (1/4) * np.abs(1 + np.exp(1j*2*phase))**2
    rho[6, 6] = (1/8) * np.abs(1 - np.exp(1j*2*phase))**2
    rho[1, 5] = 1j*sqrt(2)/8 * (1 - np.exp(1j*2*phase)) * (1 + np.exp(-1j*2*phase))
    rho[5, 1] = 1j*sqrt(2)/8 * (1 + np.exp(1j*2*phase)) * (1 - np.exp(-1j*2*phase))
    rho[8, 0] = (1/8) * (np.exp(1j*2*phase) - 1) * (1 - np.exp(-1j*2*phase))
    rho[0, 8] = (1/8) * (1 - np.exp(1j*2*phase)) * (np.exp(-1j*2*phase) - 1)
    rho[3, 7] = 1j*sqrt(2)/8 * (np.exp(1j*2*phase) + 1) * (np.exp(-1j*2*phase) - 1)
    rho[7, 3] = 1j*sqrt(2)/8 * (np.exp(1j*2*phase) - 1) * (np.exp(-1j*2*phase) + 1)
    return rho

num = 100

phase_arr = np.linspace(0, np.pi, num)

negat_array = np.zeros(num)

for i in range(num):
    matr = dens_matrix(phase_arr[i])
    w, v = np.linalg.eig(matr)
    neg = 0
    for eigval in w:
        if np.real(eigval) < 0:
            neg = neg + np.abs(np.real(eigval))
    negat_array[i] = np.log2(2 * neg + 1)


plt.plot(phase_arr, negat_array)
plt.show()
plt.xlim([0, np.pi])
plt.ylim([0, 1.05])


fn_entropy = lambda fi: - (np.sin(fi)**2 * np.log(0.5 * np.sin(fi)**2) + np.cos(fi)**2 * np.log(np.cos(fi)**2))

fn_entropy_arr = np.zeros(num)

for i in range(num):
    fn_entropy_arr[i] = fn_entropy(phase_arr[i])


plt.plot(phase_arr, fn_entropy_arr, label=r'$Log. entropy$')
plt.plot(phase_arr, negat_array, label=r'$Log. negativity$')
plt.xlim([0, np.pi])
plt.ylim([0, 1.2])
plt.legend()
plt.grid(True)
#my_xticks = ['0','1']
#plt.yticks(phase_arr, my_xticks)
plt.set_xticks([0, np.pi])
plt.set_xticklabels(['0', '$\pi$'])
plt.show()



fig, ax = plt.subplots()
ax.plot(phase_arr, fn_entropy_arr, 'b', label=r'$Log. entropy$')
ax.plot(phase_arr, negat_array, 'r--', label=r'$Log. negativity$')
# set ticks and tick labels
ax.set_xlim((0, np.pi))
ax.set_xticks([0, 0.5*np.pi, np.pi])
ax.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
ax.set_ylim((0, np.log(3) + 0.05))
ax.set_yticks([0, fn_entropy_arr[int(num/2)], 1, np.log(3)])
ax.set_yticklabels(['0', '0.69', '1', '$\ln{(3)}$'])

#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

plt.grid(True)
plt.legend()

plt.show()





