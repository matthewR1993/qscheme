import numpy as np
import qutip as qt
from math import sqrt
import matplotlib.pyplot as plt
from customutils import *
from state_configurations import single_photon, plot_state

# Scheme has only two channels in first area
# four in the middle and two in the end.

# Set up default scheme parameters.
# These r and t are from 0 to 1. Condition: t^2 + r^2 = 1
t1 = sqrt(0.5)
r1 = sqrt(0.5)

t2 = sqrt(0.5)
r2 = sqrt(0.5)

t3 = sqrt(0.5)
r3 = sqrt(0.5)

t4 = sqrt(0.5)
r4 = sqrt(0.5)

# can be set small for simple configurations
series_length = 20


# set up input state as a Taylor series
input_st = single_photon(series_length)
# plot_state(input_st)

# set up auxiliary state as a Taylor series
auxiliary_st = single_photon(series_length)

# Calculating state after first BS.






