import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum


L = 10


# INPUT
# state1 = single_photon(ser_len)
state1 = coherent_state(L, alpha=1.0)

# AUXILIARY
# state2 = coherent_state(L, alpha=1.0)
state2 = single_photon(L)

I = tf.constant(1j, tf.complex128)

st1_tf = tf.constant(state1, tf.complex128)
st2_tf = tf.constant(state2, tf.complex128)

# Unapplied state
mut_state = tf.tensordot(
    st1_tf,
    st2_tf,
    axes=0,
    name=None
)

T1 = tf.Variable(0.7, trainable=True, dtype=tf.float64)
# R1 = tf.Variable(0.5, trainable=True)


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
                    fact_const[m, n, k, l] = (1j) ** (k + l) * factorial(m) * factorial(n) / (
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

    state_out = tf.reduce_sum(d, axis=[2, 3, 4, 5])
    return state_out


state_out = bs_transformation_tf(mut_state, T1)

state_trace = tf.trace(tf.abs(state_out))
J = tf.cast(state_trace, tf.float64)


# Minimizer
# with tf.Session() as sess:
#     opt = tf.train.GradientDescentOptimizer(0.01)
#     opt_op = opt.minimize(J, var_list=[T1])
#     opt_op.run()


# WORKS.
opt = tf.train.GradientDescentOptimizer(0.0005)
grads_and_vars = opt.compute_gradients(J, [T1])
train = opt.apply_gradients(grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(400):
    print(sess.run([T1]))
    sess.run(train)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # m1_arr = m1.eval()
    # m2_arr = m2.eval()
    # t1_tf.eval()
    # t_pows_tf.eval()
    # d = tf.sqrt(T1).eval()
    #r1_tf.eval()

    state_out_val = state_out.eval()
    J_val = J.eval()


plt.imshow(np.abs(state_out_val))
plt.show()
