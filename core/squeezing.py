import numpy as np
from math import sqrt


# Takes an applied density matrix for 2 channels.
# Returns:
# Average value of coordinate quadrature
# <X> = <(a + conj(a))/2>
def coord_aver(dm, channel):
    size = len(dm)
    if channel is 1:
        sum1 = 0
        for m in range(1, size):
            for n in range(size):
                sum1 = sum1 + sqrt(m) * dm[m, n, m - 1, n]
        sum2 = 0
        for m in range(size - 1):
            for n in range(size):
                sum2 = sum2 + sqrt(m + 1) * dm[m, n, m + 1, n]
    elif channel is 2:
        sum1 = 0
        for m in range(size):
            for n in range(1, size):
                sum1 = sum1 + sqrt(n) * dm[m, n, m, n - 1]
        sum2 = 0
        for m in range(size):
            for n in range(size - 1):
                sum2 = sum2 + sqrt(n + 1) * dm[m, n, m, n + 1]
    else:
        raise ValueError('Wrong configuration')
    dx = 0.5 * (sum1 + sum2)
    return dx


# Takes an applied density matrix for 2 channels.
# Returns:
# Average value of coordinate quadrature
# <P> = <(a - conj(a))/2j>
def impulse_aver(dm, channel):
    size = len(dm)
    if channel is 1:
        sum1 = 0
        for m in range(1, size):
            for n in range(size):
                sum1 = sum1 + sqrt(m) * dm[m, n, m - 1, n]
        sum2 = 0
        for m in range(size - 1):
            for n in range(size):
                sum2 = sum2 + sqrt(m + 1) * dm[m, n, m + 1, n]
    elif channel is 2:
        sum1 = 0
        for m in range(size):
            for n in range(1, size):
                sum1 = sum1 + sqrt(n) * dm[m, n, m, n - 1]
        sum2 = 0
        for m in range(size):
            for n in range(size - 1):
                sum2 = sum2 + sqrt(n + 1) * dm[m, n, m, n + 1]
    else:
        raise ValueError('Wrong configuration')
    dp = (1 / 2j) * (sum1 - sum2)
    return dp


# Takes an applied density matrix for 2 channels.
# Returns:
# Average value of coordinate quadrature product of two channels
# <X1*X2> = (1/4) * <(a1 + conj(a1))*(a2 + conj(a2))>
def prod_coord_aver(dm):
    size = len(dm)
    # a1 * a2
    sum1 = 0
    for m in range(1, size):
        for n in range(1, size):
            sum1 = sum1 + sqrt(m * n) * dm[m, n, m - 1, n - 1]
    # a1 * conj(a2)
    sum2 = 0
    for m in range(1, size):
        for n in range(size - 1):
            sum2 = sum2 + sqrt(m * (n + 1)) * dm[m, n, m - 1, n + 1]
    # conj(a1) * a2
    sum3 = 0
    for m in range(size - 1):
        for n in range(1, size):
            sum3 = sum3 + sqrt((m + 1) * n) * dm[m, n, m + 1, n - 1]
    # conj(a1) * conj(a2)
    sum4 = 0
    for m in range(size - 1):
        for n in range(size - 1):
            sum4 = sum4 + sqrt((m + 1) * (n + 1)) * dm[m, n, m + 1, n + 1]
    d = 0.25 * (sum1 + sum2 + sum3 + sum4)
    return d


# Takes an applied density matrix for 2 channels.
# Returns:
# Average value of coordinate quadrature product of two channels
# <P1*P2> = (-1/4) * <(a1 - conj(a1))*(a2 - conj(a2))>
def prod_impulse_aver(dm):
    size = len(dm)
    # a1 * a2
    sum1 = 0
    for m in range(1, size):
        for n in range(1, size):
            sum1 = sum1 + sqrt(m * n) * dm[m, n, m - 1, n - 1]
    # a1 * conj(a2)
    sum2 = 0
    for m in range(1, size):
        for n in range(size - 1):
            sum2 = sum2 + sqrt(m * (n + 1)) * dm[m, n, m - 1, n + 1]
    # conj(a1) * a2
    sum3 = 0
    for m in range(size - 1):
        for n in range(1, size):
            sum3 = sum3 + sqrt((m + 1) * n) * dm[m, n, m + 1, n - 1]
    # conj(a1) * conj(a2)
    sum4 = 0
    for m in range(size - 1):
        for n in range(size - 1):
            sum4 = sum4 + sqrt((m + 1) * (n + 1)) * dm[m, n, m + 1, n + 1]
    d = - 0.25 * (sum1 - sum2 - sum3 + sum4)
    return d


# Takes an applied density matrix for 2 channels.
# Returns:
# Average value of square coordinate quadrature of specific channel
# <X^2> = (1/4) * <(a + conj(a))^2>
def coord_square_aver(dm, channel):
    size = len(dm)
    if channel is 1:
        # -1 + 2*a*conj(a)
        sum1 = 0
        for m in range(size):
            for n in range(size):
                sum1 = sum1 + (2*(m + 1) - 1) * dm[m, n, m, n]
        # a^2
        sum2 = 0
        for m in range(2, size):
            for n in range(size):
                sum2 = sum2 + sqrt(m*(m - 1)) * dm[m, n, m - 2, n]
        # conj(a)^2
        sum3 = 0
        for m in range(size - 2):
            for n in range(size):
                sum3 = sum3 + sqrt((m + 1)*(m + 2)) * dm[m, n, m + 2, n]
    elif channel is 2:
        # -1 + 2*a*conj(a)
        sum1 = 0
        for m in range(size):
            for n in range(size):
                sum1 = sum1 + (2*(n + 1) - 1) * dm[m, n, m, n]
        # a^2
        sum2 = 0
        for m in range(size):
            for n in range(2, size):
                sum2 = sum2 + sqrt(n*(n - 1)) * dm[m, n, m, n - 2]
        # conj(a)^2
        sum3 = 0
        for m in range(size):
            for n in range(size - 2):
                sum3 = sum3 + sqrt((n + 1)*(n + 2)) * dm[m, n, m, n + 2]
    else:
        raise ValueError('Wrong configuration')
    dx2 = 0.25 * (sum1 + sum2 + sum3)
    return dx2


# Takes an applied density matrix for 2 channels.
# Returns:
# Average value of square impulse quadrature of specific channel
# <P^2> = (-1/4) * <(a - conj(a))^2>
def impulse_square_aver(dm, channel):
    size = len(dm)
    if channel is 1:
        # -1 + 2*a*conj(a)
        sum1 = 0
        for m in range(size):
            for n in range(size):
                sum1 = sum1 + (2*(m + 1) - 1) * dm[m, n, m, n]
        # a^2
        sum2 = 0
        for m in range(2, size):
            for n in range(size):
                sum2 = sum2 + sqrt(m*(m - 1)) * dm[m, n, m - 2, n]
        # conj(a)^2
        sum3 = 0
        for m in range(size - 2):
            for n in range(size):
                sum3 = sum3 + sqrt((m + 1)*(m + 2)) * dm[m, n, m + 2, n]
    elif channel is 2:
        # -1 + 2*a*conj(a)
        sum1 = 0
        for m in range(size):
            for n in range(size):
                sum1 = sum1 + (2*(n + 1) - 1) * dm[m, n, m, n]
        # a^2
        sum2 = 0
        for m in range(size):
            for n in range(2, size):
                sum2 = sum2 + sqrt(n*(n - 1)) * dm[m, n, m, n - 2]
        # conj(a)^2
        sum3 = 0
        for m in range(size):
            for n in range(size - 2):
                sum3 = sum3 + sqrt((n + 1)*(n + 2)) * dm[m, n, m, n + 2]
    else:
        raise ValueError('Wrong configuration')
    dp2 = - 0.25 * (- sum1 + sum2 + sum3)
    return dp2


# Takes an applied density matrix for 2 channels.
# Returns coordinate and impuls quadratures for a chosen channel:
# D[X] = sqrt(<X^2> - (<X>)^2)
# D[P] = sqrt(<P^2> - (<P>)^2)
def squeezing_quadratures(dm, channel):
    dx = np.sqrt(coord_square_aver(dm, channel) - coord_aver(dm, channel)**2)
    dp = np.sqrt(impulse_square_aver(dm, channel) - impulse_aver(dm, channel)**2)
    return dx, dp


# Takes an applied density matrix for 2 channels.
# Returns two modes squeezing EPR correlations:
# corX = D[(X_2 - X_1)],
# corP = D[(P_1 + P_2)]
# Where D[A] is a dispersion of A: D[A] = <A^2> - (<A>)^2
def erp_squeezing_correlations(dm):
    cor_x = np.sqrt(coord_square_aver(dm, 1) - 2 * prod_coord_aver(dm) + coord_square_aver(dm, 2) - coord_aver(dm, 1)**2 + 2 * coord_aver(dm, 1) * coord_aver(dm, 2) - coord_aver(dm, 2)**2)
    cor_p = np.sqrt(impulse_square_aver(dm, 1) + 2 * prod_impulse_aver(dm) + impulse_square_aver(dm, 2) - impulse_aver(dm, 1)**2 - 2 * impulse_aver(dm, 1) * impulse_aver(dm, 2) - impulse_aver(dm, 2)**2)
    return cor_x, cor_p
