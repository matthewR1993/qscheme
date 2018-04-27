from math import sqrt


# Default scheme parameters

# Set up default scheme parameters.
# These r,t,a are from 0 to 1. Beam splitter with absorption: t^2 + r^2 + a^2 = 1
a1 = 0
t1 = sqrt(0.5)
r1 = sqrt(1 - pow(t1, 2) - pow(a1, 2))

a2 = 0
t2 = sqrt(0.5)
r2 = sqrt(1 - pow(t2, 2) - pow(a2, 2))

a3 = 0
t3 = sqrt(0.5)
r3 = sqrt(1 - pow(t3, 2) - pow(a3, 2))

a4 = 0
t4 = sqrt(0.5)
r4 = sqrt(1 - pow(t4, 2) - pow(a4, 2))

# Types of detector
DET_TYPE = 'IDEAL'
# DET_TYPE = 'REAL'

SPD = 1
