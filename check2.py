# Two single photons and two beam splitters.
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


net = 400


a20 = lambda t, fi: 1/sqrt(2) * (t**2*np.exp(1j*2*fi) + t**2 - 1)
a02 = lambda t, fi: 1/sqrt(2) * ((t**2 - 1)*np.exp(1j*2*fi) + t**2)
a11 = lambda t, fi: 1j * t2 * sqrt(1 - t2**2) * (np.exp(1j*2*fi) + 1)


t2_arr = np.linspace(0, 1, net)
phi_arr = np.linspace(0, np.pi, net)


# fixed BS 50-50, varying phase
# t2 = 1/sqrt(2)
t2 = 0.6

log_neg_arr = np.zeros(net)
fn_entr_arr = np.zeros(net)

for i in range(net):
    phi = phi_arr[i]

    log_neg = np.log2(2*(np.abs(a02(t2,phi))*np.abs(a11(t2,phi)) + np.abs(a02(t2,phi))*np.abs(a20(t2,phi)) + np.abs(a11(t2,phi))*np.abs(a20(t2,phi))) + 1)

    fn_entropy = - np.abs(a02(t2,phi))**2*np.log(np.abs(a02(t2,phi))**2) - np.abs(a20(t2,phi))**2*np.log(np.abs(a20(t2,phi))**2) - np.abs(a11(t2,phi))**2*np.log(np.abs(a11(t2,phi))**2)

    log_neg_arr[i] = log_neg

    fn_entr_arr[i] = fn_entropy


fig, ax = plt.subplots()
ax.plot(phi_arr, fn_entr_arr, label=r'$Log. FN \ entropy$')
ax.plot(phi_arr, log_neg_arr, label=r'$Log. negativity$')
ax.set_xlim((0, np.pi))
ax.set_ylim((0, 1.75))
ax.set_xticks([0, 0.5*np.pi, np.pi])
ax.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
plt.grid(True)
plt.legend()
plt.xlabel('Phase')
plt.xlabel('Entanglement')
plt.title('T2={0}'.format(str(t2**2)[:4]))
plt.show()


# phase is fixed, varying T2
ph_inpi = 0.75
phi = ph_inpi * np.pi

log_neg_arr2 = np.zeros(net)
fn_entr_arr2 = np.zeros(net)

for i in range(net):
    t2 = t2_arr[i]

    log_neg = np.log2(2*(np.abs(a02(t2,phi))*np.abs(a11(t2,phi)) + np.abs(a02(t2,phi))*np.abs(a20(t2,phi)) + np.abs(a11(t2,phi))*np.abs(a20(t2,phi))) + 1)

    fn_entropy = - np.abs(a02(t2,phi))**2*np.log(np.abs(a02(t2,phi))**2) - np.abs(a20(t2,phi))**2*np.log(np.abs(a20(t2,phi))**2) - np.abs(a11(t2,phi))**2*np.log(np.abs(a11(t2,phi))**2)

    log_neg_arr2[i] = log_neg

    fn_entr_arr2[i] = fn_entropy


plt.plot(np.square(t2_arr), fn_entr_arr2, label=r'$Log. FN \ entropy$')
plt.plot(np.square(t2_arr), log_neg_arr2, label=r'$Log. negativity$')
plt.legend()
plt.grid(True)
plt.title('Phase = {0}pi'.format(ph_inpi))
plt.xlabel('$T_{2}$')
plt.ylabel('$Entanglement$')
plt.xlim([0, 1])
plt.show()

