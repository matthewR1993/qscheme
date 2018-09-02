import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/usr/local/lib/python3.5/dist-packages')

from customutils.utils import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from core.gradient_methods import gd_with_momentum


# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

# Building a mutual state via tensor product, that returns numpy array.
mut_state_unappl = np.tensordot(input_st, auxiliary_st, axes=0)

# The phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

start_point = {
    't1': sqrt(0.5),
    't4': sqrt(0.5),
    't2': sqrt(0.5),
    't3': sqrt(0.5),
}


algo_params = {
    'alpha': 3e-2,
    'alpha_scale': 1.2,
    'betta': 0.999,
    'target_prec': 1e-3,
    'search_iter_max': 40,
    'start_point': start_point,

}

funct_params = {
    'free_par_keys': ['t1', 't4'],
    'target_quantity_min': 'EPR_X',
    'det_event': DET_CONF,
    'phase': 0.0,
    'input_state': mut_state_unappl
}

result = gd_with_momentum(algo_params=algo_params, funct_params=funct_params)

# print(result)
