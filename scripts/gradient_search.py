import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/usr/local/lib/python3.5/dist-packages')

import tensorflow as tf

from customutils.utils import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from core.gradient_methods import gradient_descent


sess = tf.Session()

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

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

# Building a mutual state via tensor product, that returns numpy array.
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)


# The phase difference before last BS
ph_inpi = 0.0
phase_diff = ph_inpi * np.pi

start_point = {
    't1': sqrt(0.5),
    'r1': sqrt(0.5),
    't4': sqrt(0.5),
    'r4': sqrt(0.5),
    't2': sqrt(0.5),
    'r2': sqrt(0.5),
    't3': sqrt(0.5),
    'r3': sqrt(0.5),
}

res = gradient_descent(
    start_point,
    mut_state_unappl,
    phase_diff,
    quantity='EPR_X',
    det_event=DET_CONF,
    delta=1e-5,
    gamma_t=1e-2
)

print(res)
