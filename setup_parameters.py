from math import sqrt


# Default scheme parameters.

# Types of detector.
DET_TYPE = 'IDEAL'
# DET_TYPE = 'REAL'

# Detector's single photon detection efficiency.
SPD = 1

# EPR variance for vacuum state, the very minimum possible.
EPR_VAR_X_VAC = sqrt(1/2)
EPR_VAR_P_VAC = sqrt(1/2)

# Varince for X and P quadratures over the vacuum state.
QUADR_VAR_X_VAC = 1/2
QUADR_VAR_P_VAC = 1/2
