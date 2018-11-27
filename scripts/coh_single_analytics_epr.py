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


alpha = 1.0
t_grd = 100

# delta(T1, T2)
t1_arr = np.linspace(0, 1, t_grd)
t2_arr = np.linspace(0, 1, t_grd)
# t2_arr = np.array([1])

phase = 1.5 * np.pi


epr_x_arr = np.zeros((t_grd, t_grd), dtype=complex)


for i in range(len(t1_arr)):
    for j in range(len(t2_arr)):
        g1 = gamma_1(t1_arr[i], t2_arr[j], phase)
        g2 = gamma_2(t1_arr[i], t2_arr[j], phase)
        g3 = gamma_3(t1_arr[i], t2_arr[j], phase)
        g4 = gamma_4(t1_arr[i], t2_arr[j], phase)
        epr_x_arr[i, j] = epr_x(alpha, g1, g2, g3, g4)

print('A real part:', np.sum(np.real(epr_x_arr)))
print('An image part:', np.sum(np.imag(epr_x_arr)))

print('Minimum:', np.amin(np.real(epr_x_arr)))
print('Maximum:', np.amax(np.real(epr_x_arr)))
# Minimum: 0.9999999999999992
# Maximum: 1.0000000000000009


plt.imshow(np.real(epr_x_arr), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.xlabel('T2')
plt.ylabel('T1')
plt.show()


# delta(T1, phase)
phase_arr = np.linspace(0, 2 * np.pi, t_grd)
t1_arr = np.linspace(0, 1, t_grd)
t2_arr = np.array([1])

epr_x_arr2 = np.zeros((t_grd, t_grd), dtype=complex)


for i in range(t_grd):
    for j in range(t_grd):
        g1 = gamma_1(t1_arr[i], t2_arr[0], phase_arr[j])
        g2 = gamma_2(t1_arr[i], t2_arr[0], phase_arr[j])
        g3 = gamma_3(t1_arr[i], t2_arr[0], phase_arr[j])
        g4 = gamma_4(t1_arr[i], t2_arr[0], phase_arr[j])
        epr_x_arr2[i, j] = epr_x(alpha, g1, g2, g3, g4)


print('Minimum:', np.amin(np.real(epr_x_arr2)))
print('Maximum:', np.amax(np.real(epr_x_arr2)))
# Minimum: 0.5000629387652862
# Maximum: 1.4999370612347138

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
print('Maximum:', np.amax(np.real(epr_x_arr3)))

# t1 = 1/sqrt(2)
# Minimum: 0.50006293616306113
# Maximum: 1.499937063836938

# t1 = 1
# Minimum: 1
# Maximum: 1

plt.imshow(np.real(epr_x_arr3), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.xlabel('phase')
plt.ylabel('T2')
plt.show()


# Varying the phase and finding the global minimum.
phase_grd = 40
phase_arr = np.linspace(0, 2 * np.pi, phase_grd)

epr_x_min_arr = np.zeros(phase_grd)
epr_x_min_ind_arr = np.zeros(phase_grd, dtype=list)

for n, phase in enumerate(phase_arr):
    print('phase:', phase / np.pi)
    epr_x_arr = np.zeros((t_grd, t_grd), dtype=complex)

    for i in range(len(t1_arr)):
        for j in range(len(t2_arr)):
            g1 = gamma_1(t1_arr[i], t2_arr[j], phase)
            g2 = gamma_2(t1_arr[i], t2_arr[j], phase)
            g3 = gamma_3(t1_arr[i], t2_arr[j], phase)
            g4 = gamma_4(t1_arr[i], t2_arr[j], phase)
            epr_x_arr[i, j] = epr_x(alpha, g1, g2, g3, g4)

    print('A real part:', np.sum(np.real(epr_x_arr)))
    print('An image part:', np.sum(np.imag(epr_x_arr)))

    print('Minimum:', np.amin(np.real(epr_x_arr)))
    print('Maximum:', np.amax(np.real(epr_x_arr)))

    epr_x_min_arr[n] = np.amin(np.real(epr_x_arr))
    epr_x_min_ind_arr[n] = list(np.unravel_index(np.argmin(epr_x_arr, axis=None), epr_x_arr.shape))
    print('Min index:', epr_x_min_ind_arr[n])


plt.plot(phase_arr / np.pi, epr_x_min_arr)
plt.show()

fl = {
    'epr_x_min': epr_x_min_arr,
    'min_index': epr_x_min_ind_arr,
    'phases': phase_arr,
    't1_arr': t1_arr,
    't2_arr': t2_arr,
}

# save_root = '/home/matvei/qscheme/results/res28/'
save_root = '/Users/matvei/PycharmProjects/qscheme/results/res28/'
fname = 'epr_x_min_vs_phase_theory'
np.save(save_root + fname, fl)


# Fixing one parameter t1 and looking for the minimum.
alpha = 1.0
t_grd = 100

t2 = sqrt(0.5)
t1_arr = np.linspace(0, 1, t_grd)

phase_grd = 100
phase_arr = np.linspace(0, 2 * np.pi, phase_grd)

epr_x_min_arr = np.zeros(phase_grd)
epr_x_min_ind_arr = np.zeros(phase_grd, dtype=list)

for n, phase in enumerate(phase_arr):
    print('phase:', phase / np.pi)
    epr_x_arr = np.zeros(t_grd, dtype=complex)

    for j in range(len(t1_arr)):
            g1 = gamma_1(t1_arr[j], t2, phase)
            g2 = gamma_2(t1_arr[j], t2, phase)
            g3 = gamma_3(t1_arr[j], t2, phase)
            g4 = gamma_4(t1_arr[j], t2, phase)
            epr_x_arr[j] = epr_x(alpha, g1, g2, g3, g4)

    print('A real part:', np.sum(np.real(epr_x_arr)))
    print('An image part:', np.sum(np.imag(epr_x_arr)))

    print('Minimum:', np.amin(np.real(epr_x_arr)))
    print('Maximum:', np.amax(np.real(epr_x_arr)))

    epr_x_min_arr[n] = np.amin(np.real(epr_x_arr))
    epr_x_min_ind_arr[n] = list(np.unravel_index(np.argmin(epr_x_arr, axis=None), epr_x_arr.shape))
    print('Min index:', epr_x_min_ind_arr[n])


# EPR plot.
plt.plot(phase_arr / np.pi, epr_x_min_arr)
plt.plot(phase_arr / np.pi, [0.5]*len(phase_arr), '-.')
plt.xlim(0, 2)
plt.grid(True)
plt.xlabel('$Phase, [\pi]$', fontsize=18)
plt.show()


t1_min_vals = np.zeros(len(epr_x_min_ind_arr))

for i, item in enumerate(epr_x_min_ind_arr):
    t1_min_vals[i] = t1_arr[item[0]]

# t2 plot.
plt.plot(phase_arr / np.pi, t1_min_vals)
plt.xlabel('$Phase, [\pi]$', fontsize=18)
plt.show()


# Different alpha.
# Doesn't depend on alpha.
alpha = 1.0
phase_grd = 30
t_grd = 80

t1_arr = np.linspace(0, 1, t_grd)
t2_arr = np.linspace(0, 1, t_grd)

phase_arr = np.linspace(0, 2 * np.pi, phase_grd)

epr_x_min_arr = np.zeros(phase_grd)

for n, phase in enumerate(phase_arr):
    print('phase:', phase / np.pi)
    epr_x_arr = np.zeros((t_grd, t_grd), dtype=complex)

    for i in range(len(t1_arr)):
        for j in range(len(t2_arr)):
            g1 = gamma_1(t1_arr[i], t2_arr[j], phase)
            g2 = gamma_2(t1_arr[i], t2_arr[j], phase)
            g3 = gamma_3(t1_arr[i], t2_arr[j], phase)
            g4 = gamma_4(t1_arr[i], t2_arr[j], phase)
            epr_x_arr[i, j] = epr_x(alpha, g1, g2, g3, g4)

    print('A real part:', np.sum(np.real(epr_x_arr)))
    print('An image part:', np.sum(np.imag(epr_x_arr)))

    print('Minimum:', np.amin(np.real(epr_x_arr)))
    print('Maximum:', np.amax(np.real(epr_x_arr)))

    epr_x_min_arr[n] = np.amin(np.real(epr_x_arr))


plt.plot(phase_arr / np.pi, epr_x_min_arr)
plt.plot(phase_arr / np.pi, [0.5]*len(phase_arr), '-.')
plt.xlim(0, 2)
plt.grid(True)
plt.xlabel('$Phase, [\pi]$', fontsize=18)
plt.show()


# Varying only the phase.
alpha = 1.0
t_grd = 200

t1 = sqrt(0.5)
t2 = sqrt(0.5)

phase_grd = 80
phase_arr = np.linspace(0, 2 * np.pi, phase_grd)

epr_x_val = np.zeros(phase_grd)

for n, phase in enumerate(phase_arr):
    print('phase:', phase / np.pi)

    g1 = gamma_1(t1, t2, phase)
    g2 = gamma_2(t1, t2, phase)
    g3 = gamma_3(t1, t2, phase)
    g4 = gamma_4(t1, t2, phase)
    epr_x_val[n] = epr_x(alpha, g1, g2, g3, g4)

    print('EPR_X:', epr_x_val[n])


# EPR plot.
plt.plot(phase_arr / np.pi, epr_x_val)
plt.plot(phase_arr / np.pi, [0.5]*len(phase_arr), '-.')
plt.xlim(0, 2)
plt.grid(True)
plt.xlabel('$Phase, [\pi]$', fontsize=18)
plt.show()
