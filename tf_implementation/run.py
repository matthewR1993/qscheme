import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum

from tf_implementation.core.squeezing import *


L = 10


# state1 = single_photon(L)
state1 = coherent_state(L, alpha=1.0)

# state2 = coherent_state(L, alpha=1.0)
state2 = single_photon(L)

st1_tf = tf.constant(state1, tf.complex128)
st2_tf = tf.constant(state2, tf.complex128)

I = tf.constant(1j, tf.complex128)


def bs_transformation_tf(input_state, T1):
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

    t1_tf = tf.pow(tf.sqrt(T1), t_pows_tf)
    r1_tf = tf.pow(tf.sqrt(tf.add(tf.negative(T1), 1)), r_pows_tf)

    m1 = tf.tensordot(input_state, tf.ones([sz, sz], tf.complex128), axes=0)
    m2 = tf.multiply(tf.cast(tf.multiply(t1_tf, r1_tf), tf.complex128), bs_const_tf)

    c1 = tf.multiply(m1, m2)
    c2 = tf.tensordot(tf.ones([sz * 2 - 1, sz * 2 - 1], tf.complex128), c1, axes=0)
    d = tf.multiply(p1, c2)

    return tf.reduce_sum(d, axis=[2, 3, 4, 5])


def phase_mod(phase, input, input_type, channel):
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


with tf.name_scope('system') as scope:
    # Unapplied input state.
    mut_state = tf.tensordot(st1_tf, st2_tf, axes=0, name='input_state')

    # Trainable parameters.
    phase = tf.Variable(0.1 * np.pi, trainable=True, dtype=tf.float64, name='phase')
    T1 = tf.Variable(0.3, trainable=True, dtype=tf.float64, name='T1')
    T2 = tf.Variable(0.3, trainable=True, dtype=tf.float64, name='T2')

    s1 = bs_transformation_tf(mut_state, T1)
    s2 = phase_mod(phase, s1[:L, :L], input_type='state', channel=2)
    state_out = bs_transformation_tf(s2, T2)

    dm_out = tf.einsum('kl,mn->klmn', state_out, tf.conj(state_out))

    # Cost function.
    # cost = tf.cast(tf.trace(tf.abs(state_out)), tf.float64, name='cost')

    # Cost function.
    cor_x, _ = erp_squeezing_correlations_tf(dm_out)
    cost = tf.cast(cor_x, tf.float64)

    # Register summaries.
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('T1', T1)
    tf.summary.scalar('T2', T2)
    tf.summary.scalar('phase', phase)


optimizer = tf.train.AdamOptimizer()
minimize_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tensorboard --logdir=/home/matvei/qscheme/tf_implementation/logs/summaries/log
# http://localhost:6006
summaries_dir = '/home/matvei/qscheme/tf_implementation/logs/summaries'
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(summaries_dir + '/log', sess.graph)

max_steps = 800
display_step = 20
summarize_step = 10

cost_progress = []

for i in range(max_steps):
    [_, summary, cost_val, T1_val, T2_val, phase_val] = sess.run([minimize_op, merged, cost, T1, T2, phase])
    cost_progress.append({'cost': cost_val, 'T1': T1_val, 'T2': T2_val, 'phase': phase_val})
    # cost_progress.append(cost_val)

    # Prints progress.
    if i % display_step == 0:
        print("Rep: {} Cost: {} T1: {} T2: {} phase: {}".format(i, cost_val, T1_val, T2_val, phase_val))
    if i % summarize_step == 0:
        writer.add_summary(summary, i)


plt.plot([c['cost'] for c in cost_progress])
plt.xlabel('cost')
plt.xlabel('step')
plt.show()

# plt.plot([c['par_value'] for c in cost_progress])
# plt.show()

# pd.DataFrame(cost_progress).plot()




# def two_bs2x4_transform(t1, r1, t2, r2, input_state):
#     """
#     Transformation at 2 beam splitters.
#     Two input channels and four output channles - 2x4 transformation.
#     Creation operators transformation:
#     a1 => t1 a2 + i r1 a1.
#     a2 => t2 a4 + i r2 a3.
#     With transmission and reflection coefficients:
#     t1^2 + r1^2 = 1.
#     t2^2 + r2^2 = 1.
#     :param t1: BS1 transmission.
#     :param r1: BS1 reflection.
#     :param t2: BS2 transmission.
#     :param r2: BS2 reflection.
#     :param input_state: Two channels(modes) unapllied state.
#     :return: Four channels(modes) unapllied state.
#     """
#     size = len(input_state)
#     output_state = np.zeros((size,) * 4, dtype=complex)
#     for m in range(size):
#         for n in range(size):
#
#             for k in range(m + 1):
#                 for l in range(n + 1):
#                     # channels indexes
#                     ind1 = k
#                     ind2 = m - k
#                     ind3 = l
#                     ind4 = n - l
#                     coeff = input_state[m, n] * t1**(m - k) * (1j*r1)**k * t2**(n - l) * (1j*r2)**l * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))
#                     output_state[ind1, ind2, ind3, ind4] = output_state[ind1, ind2, ind3, ind4] + coeff
#
#     return output_state

# L = 10
# dm = tf.constant(np.random.rand(L, L, L, L), tf.complex128)
#
# sess = tf.Session()
#
# res = erp_squeezing_correlations_tf(dm)
# print(res[0].eval(session=sess), res[1].eval(session=sess))
#
# print(erp_squeezing_correlations(dm.eval(session=sess)))


# Detection:
# def detection(state, type):
#     return 0


# def bs_2x4_transform(T1, T2, input_state):
#     return 0

