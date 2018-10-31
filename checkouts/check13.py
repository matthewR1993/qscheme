# A coherent state plus a single photon, epr variance.
# 1) alpha=1, phase=?, vary: t1, t2
# 2) alpha=1, t2=1, vary: t1, phase
# 3) alpha=1, t2=1/sqrt(2), vary: t1, phase
# 4) alpha=1, t1=1, vary: t2, phase
# 5) alpha=1, t1=1/sqrt(2), vary: t2, phase

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon


# Parameters for states
series_length = 8
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
input_st = single_photon(series_length)
# AUXILIARY - the state in the second(on top) channel
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)


# 1)
phase = 0.0 * np.pi

input_state = np.tensordot(input_st, auxiliary_st, axes=0)

t1_grid = 40
t2_grid = 40


t1_arr = np.linspace(0, 1, t1_grid)
t2_arr = np.linspace(0, 1, t2_grid)

sz = (t1_grid, t2_grid)
epr_correl_x = np.zeros(sz, dtype=complex)
epr_correl_p = np.zeros(sz, dtype=complex)


for n1 in range(t1_grid):
    for n2 in range(t2_grid):
        print('n1, n2:', n1, n2)
        t1 = t1_arr[n1]
        r1 = sqrt(1 - t1**2)
        t2 = t2_arr[n2]
        r2 = sqrt(1 - t2**2)
        # BS1.
        state1_unappl = bs2x2_transform(t1, r1, input_state)
        # Phase modulation.
        state2_unappl = phase_modulation_state(state1_unappl, phase)
        # BS2.
        state3_unappl = bs2x2_transform(t2, r2, state2_unappl)

        # Form a density matrix. It is applied.
        dm = dens_matrix(make_state_appliable(state3_unappl))

        epr_x, epr_p = erp_squeezing_correlations(dm)
        epr_correl_x[n1, n2] = epr_x

