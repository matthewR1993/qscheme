import numpy as np
from math import sqrt


def coord_aver(dm, channel):
    '''
    Average value of a coordinate quadrature:
    <X> = <(a + conj(a))/2>
    :param dm: Applied density matrix in 2 channels.
    :param channel: Number of the channel.
    :return: Average value of coordinate quadrature: <X> = <(a + conj(a))/2>
    '''
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


def impulse_aver(dm, channel):
    '''
    The average value of an impulse quadrature:
    <P> = <(a - conj(a))/2j>
    :param dm: Applied density matrix in 2 channels.
    :param channel: Number of the channel.
    :return: Average value of an impulse quadrature: <P> = <(a - conj(a))/2j>
    '''
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


def prod_coord_aver(dm):
    '''
    Average value of coordinate quadrature product in 2 channels.
    :param dm: Applied density matrix in 2 channels.
    :return: Average value of coordinate quadrature product in 2 channels:
    <X1*X2> = (1/4) * <(a1 + conj(a1))*(a2 + conj(a2))>
    '''
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


def prod_impulse_aver(dm):
    '''
    Average value of coordinate quadrature product in 2 channels.
    :param dm: Applied density matrix in 2 channels.
    :return:  Average value of coordinate quadrature product in 2 channels:
    <P1*P2> = (-1/4) * <(a1 - conj(a1))*(a2 - conj(a2))>
    '''
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


def coord_square_aver(dm, channel):
    '''
    Average value of the square coordinate quadrature for a specific channel.
    :param dm: Applied density matrix in 2 channels
    :param channel: Number of the channel.
    :return: Average value of the square coordinate quadrature for a specific channel:
    <X^2> = (1/4) * <(a + conj(a))^2>
    '''
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


def impulse_square_aver(dm, channel):
    '''
    An average value of the square impulse quadrature for a specific channel.
    :param dm: Applied density matrix for 2 channels
    :param channel: Number of the channel.
    :return: Average value of the square impulse quadrature for a specific channel:
    <P^2> = (-1/4) * <(a - conj(a))^2>
    '''
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


def squeezing_quadratures(dm, channel):
    '''
    Coordinate and impulse quadratures for a chosen channel:
    :param dm: Applied density matrix for 2 channels
    :param channel: Number of the channel.
    :return: Coordinate and impulse quadratures for a chosen channel:
    D[X] = sqrt(<X^2> - (<X>)^2)
    D[P] = sqrt(<P^2> - (<P>)^2)
    '''
    dx = np.sqrt(coord_square_aver(dm, channel) - coord_aver(dm, channel)**2)
    dp = np.sqrt(impulse_square_aver(dm, channel) - impulse_aver(dm, channel)**2)
    return dx, dp


def erp_squeezing_correlations(dm):
    '''
    Two modes squeezing EPR correlations.
    :param dm: Applied density matrix for 2 channels
    :return: EPR operator's variances:
    corX = var[(X_2 - X_1)]
    corP = var[(P_1 + P_2)]
    Where var[A] is a variance of A: var[A] = <A^2> - (<A>)^2
    '''
    cor_x = coord_square_aver(dm, 1) - 2 * prod_coord_aver(dm) + coord_square_aver(dm, 2) - coord_aver(dm, 1)**2 + 2 * coord_aver(dm, 1) * coord_aver(dm, 2) - coord_aver(dm, 2)**2
    cor_p = impulse_square_aver(dm, 1) + 2 * prod_impulse_aver(dm) + impulse_square_aver(dm, 2) - impulse_aver(dm, 1)**2 - 2 * impulse_aver(dm, 1) * impulse_aver(dm, 2) - impulse_aver(dm, 2)**2
    return cor_x, cor_p
