import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
from math import sqrt


def gamma_1(t1, t2, phi):
    return 1j * t1 * np.exp(1j * phi) * sqrt(1 - t2**2) + 1j * sqrt(1 - t1**2) * t2


def gamma_2(t1, t2, phi):
    return t1 * t2 * np.exp(1j * phi) - sqrt(1 - t1**2) * sqrt(1 - t2**2)


def gamma_3(t1, t2, phi):
    return t1 * t2 - sqrt(1 - t1**2) * sqrt(1 - t2**2) * np.exp(1j * phi)


def gamma_4(t1, t2, phi):
    return 1j * t1 * sqrt(1 - t2**2) + 1j * t2 * sqrt(1 - t1**2) * np.exp(1j * phi)


# A constant C with the wave.
def c_sl(alpha, g1, g2):
    return np.exp(0.5 * np.abs(alpha)**2 * (np.abs(g1)**2 + np.abs(g2)**2 - 1))


def epr_x(alpha, g1, g2, g3, g4):
    b1 = alpha * g1
    b2 = alpha * g2

    # Squared constant C with the wave.
    c2 = c_sl(alpha, g1, g2)**2

    # <a1_conj>
    a1_c = c2 * (
        np.abs(g3)**2 * (np.conj(b1) * (2 + np.abs(b1)**2)) +
        np.conj(g3) * g4 * np.conj(b2) * (1 + np.abs(b1)**2) +
        np.conj(g4) * g3 * b2 * (np.conj(b1)**2) +
        np.abs(g4)**2 * np.conj(b1) * (1 + np.abs(b2)**2)
    )

    # <a2_conj>
    a2_c = c2 * (
        np.abs(g3)**2 * np.conj(b2) * (1 + np.abs(b1)**2) +
        np.conj(g3) * g4 * b1 * (np.conj(b2)**2) +
        np.conj(g4) * g3 * np.conj(b1) * (1 + np.abs(b2)**2) +
        np.abs(g4)**2 * np.conj(b2) * (2 + np.abs(b2)**2)
    )

    # <a1>
    a1 = c2 * (
        np.abs(g3)**2 * b1 * (2 + np.abs(b1)**2) +
        np.conj(g3) * g4 * np.conj(b2) * b1**2 +
        np.conj(g4) * g3 * b2 * (1 + np.abs(b1)**2) +
        np.abs(g4)**2 * b1 * (1 + np.abs(b2)**2)
    )

    # <a2>
    a2 = c2 * (
        np.abs(g3)**2 * b2 * (1 + np.abs(b1)**2) +
        np.conj(g3) * g4 * b1 * (1 + np.abs(b2)**2) +
        np.conj(g4) * g3 * np.conj(b1) * b2**2 +
        np.abs(g4)**2 * b2 * (2 + np.abs(b2)**2)
    )

    # Quadratic elements:
    # <(a1)^2>
    a1quadr = c2 * (
        np.abs(g3)**2 * b1**2 * (3 + np.abs(b1)**2) +
        np.conj(g3) * g4 * b1**3 * np.conj(b2) +
        np.conj(g4) * g3 * b1 * b2 * (2 + np.abs(b1)**2) +
        np.abs(g4)**2 * b1**2 * (1 + np.abs(b2)**2)
    )

    # <a1_conj * a1>
    a1conj_a1 = c2 * (
        np.abs(g3)**2 * (np.abs(b1)**4 + 3*np.abs(b1)**2 + 1) +
        np.conj(g3) * g4 * b1 * np.conj(b2) * (np.abs(b1)**2 + 1) +
        np.conj(g4) * g3 * b2 * np.conj(b1) * (np.abs(b1)**2 + 1) +
        np.abs(g4)**2 * np.abs(b1)**2 * (np.abs(b2)**2 + 1)
    )

    # <(a1_conj)^2>
    a1conj_quadr = c2 * (
        np.abs(g3)**2 * np.conj(b1)**2 * (3 + np.abs(b1)**2) +
        np.conj(g3) * g4 * np.conj(b1) * np.conj(b2) * (2 + np.abs(b1)**2) +
        np.conj(g4) * g3 * np.conj(b1)**3 * b2 +
        np.abs(g4)**2 * np.conj(b1)**2 * (1 + np.abs(b2)**2)
    )

    # <(a2)^2>
    a2quadr = c2 * (
        np.abs(g3)**2 * b2**2 * (1 + np.abs(b1)**2) +
        np.conj(g3) * g4 * b1 * b2 * (2 + np.abs(b2)**2) +
        np.conj(g4) * g3 * b2**3 * np.conj(b1) +
        np.abs(g4)**2 * b2**2 * (np.abs(b2)**2 + 3)
    )

    # <a2_conj * a2>
    a2conj_a2 = c2 * (
        np.abs(g3)**2 * np.abs(b2)**2 * (np.abs(b1)**2 + 1) +
        np.conj(g3) * g4 * b1 * np.conj(b2) * (np.abs(b2)**2 + 1) +
        np.conj(g4) * g3 * np.conj(b1) * b2 * (np.abs(b2)**2 + 1) +
        np.abs(g4)**2 * (np.abs(b2)**4 + 3*np.abs(b2)**2 + 1)
    )

    # <(a2_conj)^2>
    a2conj_quadr = c2 * (
        np.abs(g3)**2 * np.conj(b2)**2 * (np.abs(b1)**2 + 1) +
        np.conj(g3) * g4 * b1 * (np.conj(b2)**3) +
        np.conj(g4) * g3 * np.conj(b1) * np.conj(b2) * (np.abs(b2)**2 + 2) +
        np.abs(g4)**2 * np.conj(b2)**2 * (np.abs(b2)**2 + 3)
    )

    # <a1 * a2>
    a1_a2 = c2 * (
        np.abs(g3)**2 * b1 * b2 * (np.abs(b1)**2 + 2) +
        np.conj(g3) * g4 * b1**2 * (np.abs(b2)**2 + 1) +
        np.conj(g4) * g3 * b2**2 * (np.abs(b1)**2 + 1) +
        np.abs(g4)**2 * b1 * b2 * (np.abs(b2)**2 + 2)
    )

    # <a1_conj * a2_conj>
    a1conj_a2conj = c2 * (
        np.abs(g3)**2 * np.conj(b1) * np.conj(b2) * (np.abs(b1)**2 + 2) +
        np.conj(g3) * g4 * np.conj(b2)**2 * (np.abs(b1)**2 + 1) +
        np.conj(g4) * g3 * np.conj(b1)**2 * (np.abs(b2)**2 + 1) +
        np.abs(g4)**2 * np.conj(b1) * np.conj(b2) * (np.abs(b2)**2 + 2)
    )

    # <a1 * a2_conj>
    a1_a2conj = c2 * (
        np.abs(g3)**2 * np.conj(b2) * b1 * (np.abs(b1)**2 + 2) +
        np.conj(g3) * g4 * b1**2 * (np.conj(b2)**2) +
        np.conj(g4) * g3 * (np.abs(b2)**2 + 1) * (np.abs(b1)**2 + 1) +
        np.abs(g4)**2 * b1 * np.conj(b2) * (np.abs(b2)**2 + 2)
    )

    # <a1_conj * a2>
    a1conj_a2 = c2 * (
        np.abs(g3)**2 * b2 * np.conj(b1) * (np.abs(b1)**2 + 2) +
        np.conj(g3) * g4 * (np.abs(b1)**2 + 1) * (np.abs(b2)**2 + 1) +
        np.conj(g4) * g3 * b2**2 * (np.conj(b1)**2) +
        np.abs(g4)**2 * np.conj(b1) * b2 * (np.abs(b2)**2 + 2)
    )

    # <X1 - X2> = 0.5 * (<a1_conj> + <a1> - <a2_conj> - <a2>)
    s1 = 0.5 * (a1_c + a1 - a2_c - a2)

    # <(X1 - X2)^2>
    s2 = (
        0.25 * (a1quadr + 2 * a1conj_a1 + 1 + a1conj_quadr) -
        0.5 * (a1_a2 + a1_a2conj + a1conj_a2 + a1conj_a2conj) +
        0.25 * (a2quadr + 2 * a2conj_a2 + 1 + a2conj_quadr)
    )

    # s2 = 0
    variance = s2 - s1**2
    return variance


t_grd = 100
t1_arr = np.linspace(0, 1, t_grd)
# t2_arr = np.linspace(0, 1, t_grd)
t2_arr = np.array([1])

phase = 0.5 * np.pi

alpha = 1.0


epr_x_arr = np.zeros((t_grd, t_grd), dtype=complex)


for i in range(t_grd):
    for j in range(t_grd):
        g1 = gamma_1(t1_arr[i], t2_arr[j], phase)
        g2 = gamma_2(t1_arr[i], t2_arr[j], phase)
        g3 = gamma_3(t1_arr[i], t2_arr[j], phase)
        g4 = gamma_4(t1_arr[i], t2_arr[j], phase)
        epr_x_arr[i, j] = epr_x(alpha, g1, g2, g3, g4)

print('A real part:', np.sum(np.real(epr_x_arr)))
print('An image part:', np.sum(np.imag(epr_x_arr)))

print('Minimum:', np.amin(np.real(epr_x_arr)))

plt.imshow(np.real(epr_x_arr), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
# plt.scatter(x=[epr_x_amin_ind[1]], y=[epr_x_amin_ind[0]], c='r', s=80, marker='+')
# plt.scatter(x=[50], y=[50], c='g', s=80, marker='+')
plt.xlabel('T2')
plt.ylabel('T1')
plt.show()


phase_arr = np.linspace(0, 2 * np.pi, t_grd)
t1_arr = np.linspace(0, 1, t_grd)
t2_arr = np.array([1/sqrt(2)])

epr_x_arr2 = np.zeros((t_grd, t_grd), dtype=complex)


for i in range(t_grd):
    for j in range(t_grd):
        g1 = gamma_1(t1_arr[i], t2_arr[0], phase_arr[j])
        g2 = gamma_2(t1_arr[i], t2_arr[0], phase_arr[j])
        g3 = gamma_3(t1_arr[i], t2_arr[0], phase_arr[j])
        g4 = gamma_4(t1_arr[i], t2_arr[0], phase_arr[j])
        epr_x_arr2[i, j] = epr_x(alpha, g1, g2, g3, g4)


print('Minimum:', np.amin(np.real(epr_x_arr2)))

plt.imshow(np.real(epr_x_arr2), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.xlabel('phase')
plt.ylabel('T1')
plt.show()


phase_arr = np.linspace(0, 2 * np.pi, t_grd)
t2_arr = np.linspace(0, 1, t_grd)
t1_arr = np.array([1/sqrt(2)])

epr_x_arr3 = np.zeros((t_grd, t_grd), dtype=complex)


for i in range(t_grd):
    for j in range(t_grd):
        g1 = gamma_1(t1_arr[0], t2_arr[i], phase_arr[j])
        g2 = gamma_2(t1_arr[0], t2_arr[i], phase_arr[j])
        g3 = gamma_3(t1_arr[0], t2_arr[i], phase_arr[j])
        g4 = gamma_4(t1_arr[0], t2_arr[i], phase_arr[j])
        epr_x_arr3[i, j] = epr_x(alpha, g1, g2, g3, g4)


print('Minimum:', np.amin(np.real(epr_x_arr3)))

plt.imshow(np.real(epr_x_arr3), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.xlabel('phase')
plt.ylabel('T2')
plt.show()
