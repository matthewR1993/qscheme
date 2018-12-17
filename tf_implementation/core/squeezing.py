import numpy as np
import tensorflow as tf
from math import sqrt


def coord_aver_tf(dm, channel):
    """
    Average value of a coordinate quadrature:
    <X> = <(a + conj(a))/2>
    :param dm: Applied density matrix in 2 channels, tf tensor.
    :param channel: Number of the channel.
    :return: Average value of coordinate quadrature: <X> = <(a + conj(a))/2>
    """
    L = dm.shape.dims[0].value
    if channel is 1:
        # s1
        m_sqrt1 = np.ones((L,)*4)
        for m in range(1, L):
            m_sqrt1[m, :, m - 1, :] = sqrt(m)

        m1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt1, tf.complex128)), shift=-1, axis=0)
        m2 = tf.transpose(m1[0:L - 1, :, 0:L - 1, :], perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m2))

        # s2
        m_sqrt2 = np.ones((L,) * 4)
        for m in range(L - 1):
            m_sqrt2[m, :, m + 1, :] = sqrt(m + 1)

        mm1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt2, tf.complex128)), shift=-1, axis=2)
        mm2 = tf.transpose(mm1[0:L - 1, :, 0:L - 1, :], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(mm2))
    elif channel is 2:
        # s1
        m_sqrt1 = np.ones((L,)*4)
        for n in range(1, L):
            m_sqrt1[:, n, :, n - 1] = sqrt(n)

        m1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt1, tf.complex128)), shift=-1, axis=1)
        m2 = tf.transpose(m1[:, 0:L - 1, :, 0:L - 1], perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m2))

        # s2
        m_sqrt2 = np.ones((L,) * 4)
        for n in range(L - 1):
            m_sqrt2[:, n, :, n + 1] = sqrt(n + 1)

        mm1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt2, tf.complex128)), shift=-1, axis=3)
        mm2 = tf.transpose(mm1[:, 0:L - 1, :, 0:L - 1], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(mm2))
    else:
        raise ValueError
    return 0.5 * tf.add(s1, s2)


def impulse_aver_tf(dm, channel):
    """
    The average value of an impulse quadrature:
    <P> = <(a - conj(a))/2j>
    :param dm: Applied density matrix in 2 channels.
    :param channel: Number of the channel.
    :return: Average value of an impulse quadrature: <P> = <(a - conj(a))/2j>
    """
    L = dm.shape.dims[0].value
    if channel is 1:
        # s1
        m_sqrt1 = np.ones((L,)*4)
        for m in range(1, L):
            m_sqrt1[m, :, m - 1, :] = sqrt(m)

        m1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt1, tf.complex128)), shift=-1, axis=0)
        m2 = tf.transpose(m1[0:L - 1, :, 0:L - 1, :], perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m2))

        # s2
        m_sqrt2 = np.ones((L,) * 4)
        for m in range(L - 1):
            m_sqrt2[m, :, m + 1, :] = sqrt(m + 1)

        mm1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt2, tf.complex128)), shift=-1, axis=2)
        mm2 = tf.transpose(mm1[0:L - 1, :, 0:L - 1, :], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(mm2))
    elif channel is 2:
        # s1
        m_sqrt1 = np.ones((L,)*4)
        for n in range(1, L):
            m_sqrt1[:, n, :, n - 1] = sqrt(n)

        m1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt1, tf.complex128)), shift=-1, axis=1)
        m2 = tf.transpose(m1[:, 0:L - 1, :, 0:L - 1], perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m2))

        # s2
        m_sqrt2 = np.ones((L,) * 4)
        for n in range(L - 1):
            m_sqrt2[:, n, :, n + 1] = sqrt(n + 1)

        mm1 = tf.roll(tf.multiply(dm, tf.constant(m_sqrt2, tf.complex128)), shift=-1, axis=3)
        mm2 = tf.transpose(mm1[:, 0:L - 1, :, 0:L - 1], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(mm2))
    else:
        raise ValueError
    dp = (1 / 2j) * tf.cast(tf.subtract(s1, s2), tf.complex128)
    return dp


def prod_coord_aver_tf(dm):
    """
    Average value of coordinate quadrature product in 2 channels.
    :param dm: Applied density matrix in 2 channels.
    :return: Average value of coordinate quadrature product in 2 channels:
    <X1*X2> = (1/4) * <(a1 + conj(a1))*(a2 + conj(a2))>
    """
    L = dm.shape.dims[0].value
    # a1 * a2
    sqrt_m1 = np.ones((L,)*4)
    for m in range(1, L):
        for n in range(1, L):
            sqrt_m1[m, n, m - 1, n - 1] = sqrt(m * n)

    m1 = tf.multiply(dm, tf.constant(sqrt_m1, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=0)
    m3 = tf.roll(m2, shift=-1, axis=1)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s1 = tf.trace(tf.trace(m4))

    # a1 * conj(a2)
    sqrt_m2 = np.ones((L,) * 4)
    for m in range(1, L):
        for n in range(L - 1):
            sqrt_m2[m, n, m - 1, n + 1] = sqrt(m * (n + 1))

    m1 = tf.multiply(dm, tf.constant(sqrt_m2, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=0)
    m3 = tf.roll(m2, shift=-1, axis=3)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s2 = tf.trace(tf.trace(m4))

    # conj(a1) * a2
    sqrt_m3 = np.ones((L,) * 4)
    for m in range(L - 1):
        for n in range(1, L):
            sqrt_m3[m, n, m + 1, n - 1] = sqrt((m + 1) * n)

    m1 = tf.multiply(dm, tf.constant(sqrt_m3, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=1)
    m3 = tf.roll(m2, shift=-1, axis=2)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s3 = tf.trace(tf.trace(m4))

    # conj(a1) * conj(a2)
    sqrt_m4 = np.ones((L,) * 4)
    for m in range(L - 1):
        for n in range(L - 1):
            sqrt_m4[m, n, m + 1, n + 1] = sqrt((m + 1) * (n + 1))

    m1 = tf.multiply(dm, tf.constant(sqrt_m4, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=2)
    m3 = tf.roll(m2, shift=-1, axis=3)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s4 = tf.trace(tf.trace(m4))
    return 0.25 * tf.add_n([s1, s2, s3, s4])


def prod_impulse_aver_tf(dm):
    """
    Average value of coordinate quadrature product in 2 channels.
    :param dm: Applied density matrix in 2 channels.
    :return:  Average value of coordinate quadrature product in 2 channels:
    <P1*P2> = (-1/4) * <(a1 - conj(a1))*(a2 - conj(a2))>
    """
    L = dm.shape.dims[0].value
    # a1 * a2
    sqrt_m1 = np.ones((L,)*4)
    for m in range(1, L):
        for n in range(1, L):
            sqrt_m1[m, n, m - 1, n - 1] = sqrt(m * n)

    m1 = tf.multiply(dm, tf.constant(sqrt_m1, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=0)
    m3 = tf.roll(m2, shift=-1, axis=1)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s1 = tf.trace(tf.trace(m4))

    # a1 * conj(a2)
    sqrt_m2 = np.ones((L,) * 4)
    for m in range(1, L):
        for n in range(L - 1):
            sqrt_m2[m, n, m - 1, n + 1] = sqrt(m * (n + 1))

    m1 = tf.multiply(dm, tf.constant(sqrt_m2, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=0)
    m3 = tf.roll(m2, shift=-1, axis=3)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s2 = tf.trace(tf.trace(m4))

    # conj(a1) * a2
    sqrt_m3 = np.ones((L,) * 4)
    for m in range(L - 1):
        for n in range(1, L):
            sqrt_m3[m, n, m + 1, n - 1] = sqrt((m + 1) * n)

    m1 = tf.multiply(dm, tf.constant(sqrt_m3, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=1)
    m3 = tf.roll(m2, shift=-1, axis=2)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s3 = tf.trace(tf.trace(m4))

    # conj(a1) * conj(a2)
    sqrt_m4 = np.ones((L,) * 4)
    for m in range(L - 1):
        for n in range(L - 1):
            sqrt_m4[m, n, m + 1, n + 1] = sqrt((m + 1) * (n + 1))

    m1 = tf.multiply(dm, tf.constant(sqrt_m4, tf.complex128))
    m2 = tf.roll(m1, shift=-1, axis=2)
    m3 = tf.roll(m2, shift=-1, axis=3)
    m4 = tf.transpose(m3[0:L - 1, 0:L - 1, 0:L - 1, 0:L - 1], perm=[0, 2, 1, 3])
    s4 = tf.trace(tf.trace(m4))
    return - 0.25 * tf.subtract(tf.subtract(tf.add(s1, s4), s2), s3)


def coord_square_aver_tf(dm, channel):
    """
    Average value of the square coordinate quadrature for a specific channel.
    :param dm: Applied density matrix in 2 channels
    :param channel: Number of the channel.
    :return: Average value of the square coordinate quadrature for a specific channel:
    <X^2> = (1/4) * <(a + conj(a))^2>
    """
    L = dm.shape.dims[0].value
    if channel is 1:
        # -1 + 2*a*conj(a)
        sqrt_m1 = np.ones((L,) * 4)
        for m in range(L):
            sqrt_m1[m, :, m, :] = 2 * (m + 1) - 1

        m1 = tf.transpose(tf.multiply(dm, tf.constant(sqrt_m1, tf.complex128)), perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m1))

        # a^2
        sqrt_m2 = np.ones((L,) * 4)
        for m in range(2, L):
            sqrt_m2[m, :, m - 2, :] = sqrt(m*(m - 1))

        m1 = tf.multiply(dm, tf.constant(sqrt_m2, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=0)
        m3 = tf.transpose(m2[0:L - 2, :, 0:L - 2, :], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(m3))

        # conj(a)^2
        sqrt_m3 = np.ones((L,) * 4)
        for m in range(L - 2):
            sqrt_m3[m, :, m + 2, :] = sqrt((m + 1)*(m + 2))

        m1 = tf.multiply(dm, tf.constant(sqrt_m3, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=2)
        m3 = tf.transpose(m2[0:L - 2, :, 0:L - 2, :], perm=[0, 2, 1, 3])
        s3 = tf.trace(tf.trace(m3))
    elif channel is 2:
        # -1 + 2*a*conj(a)
        sqrt_m1 = np.ones((L,) * 4)
        for n in range(L):
            sqrt_m1[:, n, :, n] = 2*(n + 1) - 1

        m1 = tf.transpose(tf.multiply(dm, tf.constant(sqrt_m1, tf.complex128)), perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m1))

        # a^2
        sqrt_m2 = np.ones((L,) * 4)
        for n in range(2, L):
            sqrt_m2[:, n, :, n - 2] = sqrt(n*(n - 1))

        m1 = tf.multiply(dm, tf.constant(sqrt_m2, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=1)
        m3 = tf.transpose(m2[:, 0:L - 2, :, 0:L - 2], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(m3))

        # conj(a)^2
        sqrt_m3 = np.ones((L,) * 4)
        for n in range(L - 2):
            sqrt_m3[:, n, :, n + 2] = sqrt((n + 1) * (n + 2))

        m1 = tf.multiply(dm, tf.constant(sqrt_m3, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=3)
        m3 = tf.transpose(m2[:, 0:L - 2, :, 0:L - 2], perm=[0, 2, 1, 3])
        s3 = tf.trace(tf.trace(m3))
    else:
        raise ValueError('Wrong configuration')
    return 0.25 * tf.add_n([s1, s2, s3])


def impulse_square_aver_tf(dm, channel):
    """
    An average value of the square impulse quadrature for a specific channel.
    :param dm: Applied density matrix for 2 channels
    :param channel: Number of the channel.
    :return: Average value of the square impulse quadrature for a specific channel:
    <P^2> = (-1/4) * <(a - conj(a))^2>
    """
    L = dm.shape.dims[0].value
    if channel is 1:
        # -1 + 2*a*conj(a)
        sqrt_m1 = np.ones((L,) * 4)
        for m in range(L):
            sqrt_m1[m, :, m, :] = 2 * (m + 1) - 1

        m1 = tf.transpose(tf.multiply(dm, tf.constant(sqrt_m1, tf.complex128)), perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m1))

        # a^2
        sqrt_m2 = np.ones((L,) * 4)
        for m in range(2, L):
            sqrt_m2[m, :, m - 2, :] = sqrt(m*(m - 1))

        m1 = tf.multiply(dm, tf.constant(sqrt_m2, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=0)
        m3 = tf.transpose(m2[0:L - 2, :, 0:L - 2, :], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(m3))

        # conj(a)^2
        sqrt_m3 = np.ones((L,) * 4)
        for m in range(L - 2):
            sqrt_m3[m, :, m + 2, :] = sqrt((m + 1)*(m + 2))

        m1 = tf.multiply(dm, tf.constant(sqrt_m3, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=2)
        m3 = tf.transpose(m2[0:L - 2, :, 0:L - 2, :], perm=[0, 2, 1, 3])
        s3 = tf.trace(tf.trace(m3))
    elif channel is 2:
        # -1 + 2*a*conj(a)
        sqrt_m1 = np.ones((L,) * 4)
        for n in range(L):
            sqrt_m1[:, n, :, n] = 2*(n + 1) - 1

        m1 = tf.transpose(tf.multiply(dm, tf.constant(sqrt_m1, tf.complex128)), perm=[0, 2, 1, 3])
        s1 = tf.trace(tf.trace(m1))

        # a^2
        sqrt_m2 = np.ones((L,) * 4)
        for n in range(2, L):
            sqrt_m2[:, n, :, n - 2] = sqrt(n*(n - 1))

        m1 = tf.multiply(dm, tf.constant(sqrt_m2, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=1)
        m3 = tf.transpose(m2[:, 0:L - 2, :, 0:L - 2], perm=[0, 2, 1, 3])
        s2 = tf.trace(tf.trace(m3))

        # conj(a)^2
        sqrt_m3 = np.ones((L,) * 4)
        for n in range(L - 2):
            sqrt_m3[:, n, :, n + 2] = sqrt((n + 1) * (n + 2))

        m1 = tf.multiply(dm, tf.constant(sqrt_m3, tf.complex128))
        m2 = tf.roll(m1, shift=-2, axis=3)
        m3 = tf.transpose(m2[:, 0:L - 2, :, 0:L - 2], perm=[0, 2, 1, 3])
        s3 = tf.trace(tf.trace(m3))
    else:
        raise ValueError('Wrong configuration')
    return - 0.25 * tf.subtract(tf.add(s2, s3), s1)


def erp_squeezing_correlations_tf(dm):
    """
    Two modes squeezing EPR correlations.
    :param dm: Applied density matrix for 2 channels
    :return: EPR operator's variances:
    corX = var[(X_2 - X_1)]
    corP = var[(P_1 + P_2)]
    Where var[A] is a variance of A: var[A] = <A^2> - (<A>)^2
    """
    cor_x = coord_square_aver_tf(dm, 1) - 2 * prod_coord_aver_tf(dm) + coord_square_aver_tf(dm, 2) - coord_aver_tf(dm, 1)**2 + 2 * coord_aver_tf(dm, 1) * coord_aver_tf(dm, 2) - coord_aver_tf(dm, 2)**2
    cor_p = impulse_square_aver_tf(dm, 1) + 2 * prod_impulse_aver_tf(dm) + impulse_square_aver_tf(dm, 2) - impulse_aver_tf(dm, 1)**2 - 2 * impulse_aver_tf(dm, 1) * impulse_aver_tf(dm, 2) - impulse_aver_tf(dm, 2)**2
    return cor_x, cor_p