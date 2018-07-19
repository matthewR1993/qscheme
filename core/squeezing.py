import cmath
import numpy as np
from math import sqrt, factorial


# TODO divide it and make it a constructor
# Two modes squeezing
# Takes applied density matrix for 2 channels
# Returns first and second quadratures for a chosen channel
def squeezing_quadratures(dm, channel):
    dx1 = 0  # dx1 = <X1^2> - (<X1>)^2
    dx2 = 0  # dx2 = <X2^2> - (<X2>)^2
    size = len(dm)

    if channel is 1:
        # Calculating parts for <X1^2> and <X2^2> regarding the first channel
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
        # Calculating parts for (<X1>)^2 and (<X2>)^2 regarding the first channel
        sum4 = 0
        for m in range(1, size):
            for n in range(size):
                sum4 = sum4 + sqrt(m) * dm[m, n, m - 1, n]
        sum5 = 0
        for m in range(size - 1):
            for n in range(size):
                sum5 = sum5 + sqrt(m + 1) * dm[m, n, m + 1, n]
        # All together:
        dx1 = np.sqrt(0.25 * (sum1 + sum2 + sum3) - 0.25 * (sum4 + sum5)**2)
        dx2 = np.sqrt((-0.25) * (- sum1 + sum2 + sum3) - (-0.25) * (sum4 - sum5)**2)

    elif channel is 2:
        # Calculating parts for <X1^2> and <X2^2> regarding the second channel
        sum1 = 0
        for m in range(size):
            for n in range(size):
                sum1 = sum1 + (2*(n + 1) - 1) * dm[m, n, m, n]
        sum2 = 0
        for m in range(size):
            for n in range(2, size):
                sum2 = sum2 + sqrt(n*(n - 1)) * dm[m, n, m, n - 2]
        sum3 = 0
        for m in range(size):
            for n in range(size - 2):
                sum3 = sum3 + sqrt((n + 1)*(n + 2)) * dm[m, n, m, n + 2]
        # Calculating parts for (<X1>)^2 and (<X2>)^2 regarding the second channel
        sum4 = 0
        for m in range(size):
            for n in range(1, size):
                sum4 = sum4 + sqrt(n) * dm[m, n, m, n - 1]
        sum5 = 0
        for m in range(size):
            for n in range(size - 1):
                sum5 = sum5 + sqrt(n + 1) * dm[m, n, m, n + 1]
        # All together:
        dx1 = np.sqrt(0.25 * (sum1 + sum2 + sum3) - 0.25 * (sum4 + sum5)**2)
        dx2 = np.sqrt((-0.25) * (- sum1 + sum2 + sum3) - (-0.25) * (sum4 - sum5)**2)

    else:
        raise ValueError('Wrong configuration')

    return dx1, dx2


# TODO EPR correlations.
# Two modes squeezing EPR correlations: D[(X1(2) - X1(1))/sqrt(2)], D[(X2(2) + X2(1))/sqrt(2)]
# Where D[A] is a dispersion of A
def squeezing_correlations(dm):
    dx = 0  # dx = D[(X1(2) - X1(1))/sqrt(2)]
    dp = 0  # dp = D[(X2(2) + X2(1))/sqrt(2)]

    return dx, dp
