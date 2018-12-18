import tensorflow as tf
import numpy as np
from math import factorial, sqrt


I = tf.constant(1j, tf.complex128)


def bs_transformation_tf(input_state, T):
    """
    Two channels (2x2) beam splitter transformation.
    With: T + R = 1.
    :param T: Transmission coefficient.
    :param input_state: Unapplied state in two channels(modes).
    :return: Transformed unapplied state in two channels(modes).
    """
    sz = input_state.shape.dims[0].value

    try:
        assert input_state.shape == tf.TensorShape([tf.Dimension(sz), tf.Dimension(sz)])
    except AssertionError:
        print('Input state dimension is invalid.')

    fact_const = np.zeros((sz,) * 4, dtype=np.complex)
    for m in range(sz):
        for n in range(sz):
            for k in range(m + 1):
                for l in range(n + 1):
                    fact_const[m, n, k, l] = 1j ** (k + l) * factorial(m) * factorial(n) / (
                                factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))

    bs_const_tf = tf.constant(fact_const, tf.complex128)

    p = np.zeros((sz * 2 - 1, sz * 2 - 1, sz, sz, sz, sz), dtype=complex)
    for m in range(sz):
        for n in range(sz):
            for k in range(m + 1):
                for l in range(n + 1):
                    d1 = m - k + l
                    d2 = n + k - l
                    p[d1, d2, m, n, k, l] = 1

    p1 = tf.constant(p, tf.complex128)

    r_pow = np.zeros((sz,) * 4, dtype=int)
    t_pow = np.zeros((sz,) * 4, dtype=int)
    for m in range(sz):
        for n in range(sz):
            for k in range(m + 1):
                for l in range(n + 1):
                    r_pow[m, n, k, l] = k + l
                    t_pow[m, n, k, l] = m - k + n - l

    r_pows_tf = tf.constant(r_pow, tf.float64)
    t_pows_tf = tf.constant(t_pow, tf.float64)

    t1_tf = tf.pow(tf.sqrt(T), t_pows_tf)
    r1_tf = tf.pow(tf.sqrt(tf.add(tf.negative(T), 1)), r_pows_tf)

    m1 = tf.tensordot(input_state, tf.ones([sz, sz], tf.complex128), axes=0)
    m2 = tf.multiply(tf.cast(tf.multiply(t1_tf, r1_tf), tf.complex128), bs_const_tf)

    c1 = tf.multiply(m1, m2)
    c2 = tf.tensordot(tf.ones([sz * 2 - 1, sz * 2 - 1], tf.complex128), c1, axes=0)
    d = tf.multiply(p1, c2)

    return tf.reduce_sum(d, axis=[2, 3, 4, 5])


def phase_mod(phase, input, input_type, channel):
    """
    :param phase: Phase difference
    :param input: State/density matrix, applied or unapllied.
    :param input_type: ['state', 'dm'] - state, density matrix.
    :param channel: A channel, where the phase modulator is located.
    :return:
    """
    sz = input.shape.dims[0].value
    if input_type == 'state':
        m = np.resize(np.arange(0, sz), (sz, sz))
        if channel == 1:
            m = m.T
        elif channel == 2:
            pass
        else:
            raise ValueError
        pow_m = tf.constant(m, tf.complex128)
        return tf.multiply(input, tf.pow(tf.exp(I * tf.cast(phase, tf.complex128)), pow_m))
    elif input_type == 'dm':
        raise NotImplementedError
    else:
        raise ValueError


def make_state_applicable(state):
    """
    Making state in 2 channels applicable.
    :param state: Uapplied state in 2 channels.
    :return: Applied state in 2 channels.
    """
    L = state.shape.dims[0].value
    m = np.ones((L, L), dtype=complex)
    for p1 in range(L):
        for p2 in range(L):
            m[p1, p2] = sqrt(factorial(p1) * factorial(p2))

    return tf.multiply(state, tf.constant(m))
