import cmath
import numpy as np
from math import sqrt, factorial


# Two modes squeezing
# Takes applied density matrix for 2 channels
# Returns first and second quadratures for a chosen channel
def squeezing_quadratures(dm, channel):
    dX1 = 0  # dX1 = <X1^2> - (<X1>)^2
    dX2 = 0  # dX2 = <X2^2> - (<X2>)^2
    size = len(dm)

    if channel is 1:
        # Calculating parts for <X1^2> and <X2^2>
        sum1 = 0
        for m in range(size):
            for n in range(size):
                sum1 = sum1 + (2*(m + 1) - 1) * dm[m, n, m, n]
        sum2 = 0
        for m in range(2, size):
            for n in range(size):
                sum2 = sum2 + sqrt(m*(m - 1)) * dm[m, n, m - 2, n]
        sum3 = 0
        for m in range(size - 2):
            for n in range(size):
                sum3 = sum3 + sqrt((m + 1)*(m + 2)) * dm[m, n, m + 2, n]
        # Calculating parts for (<X1>)^2 and (<X2>)^2
        sum4 = 0
        for m in range(1, size):
            for n in range(size):
                sum4 = sum4 + sqrt(m) * dm[m, n, m - 1, n]
        sum5 = 0
        for m in range(size - 1):
            for n in range(size):
                sum5 = sum5 + sqrt(m + 1) * dm[m, n, m + 1, n]
        # All together:
        dX1 = np.sqrt(0.25 * (sum1 + sum2 + sum3) - 0.25 * (sum4 + sum5)**2)
        dX2 = np.sqrt((-0.25) * (- sum1 + sum2 + sum3) - (-0.25) * (sum4 - sum5)**2)

    elif channel is 2:
        raise ValueError('Not implemented')
    else:
        raise ValueError('Wrong configuration')

    return dX1, dX2
