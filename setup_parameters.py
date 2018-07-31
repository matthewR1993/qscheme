from math import sqrt


# Default scheme parameters

# Set up default scheme parameters.
# These r,t,a are from 0 to 1. The ideal beam splitter without absorption: t^2 + r^2 = 1
t1 = sqrt(0.5)
r1 = sqrt(1 - pow(t1, 2))

t2 = sqrt(0.5)
r2 = sqrt(1 - pow(t2, 2))

t3 = sqrt(0.5)
r3 = sqrt(1 - pow(t3, 2))

t4 = sqrt(0.5)
r4 = sqrt(1 - pow(t4, 2))

# Types of detector
DET_TYPE = 'IDEAL'
# DET_TYPE = 'REAL'

# Detector's single photon detection efficiency
SPD = 1

# EPR variance for vacuum state, the very minimum possible.
EPR_VAR_X_VAC = sqrt(1/2)
EPR_VAR_P_VAC = sqrt(1/2)

# Varince for X and P quadratures over the vacuum state.
QUADR_VAR_X_VAC = 1/2
QUADR_VAR_P_VAC = 1/2
