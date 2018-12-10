import numpy as np
import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum


L = 10


# INPUT
# state1 = single_photon(ser_len)
state1 = coherent_state(L, alpha=0.7)

# AUXILIARY
state2 = coherent_state(L, alpha=0.7)
# state2 = single_photon(ser_len)

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

T1 = tf.Variable(0.5, trainable=True, dtype=tf.float64)
# R1 = tf.Variable(0.5, trainable=True)

# BS transformation.
# state2 = tf.add_n()
#
#
# def obj_func(s):
#     return 0
#
#
# J = obj_func(state2)
#
# opt = tf.train.GradientDescentOptimizer(0.01)
# grads_and_vars = opt.compute_gradients(J, [T1])


fact_const = np.zeros((L,)*4, dtype=np.complex)
for m in range(L):
    for n in range(L):
        for k in range(m + 1):
            for l in range(n + 1):
                fact_const[m, n, k, l] = (1j)**(k + l) * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))

bs_const_tf = tf.constant(fact_const, tf.complex128)

p = np.zeros((L*2 - 1, L*2 - 1, L, L, L, L), dtype=complex)
for m in range(L):
    for n in range(L):
        for k in range(m + 1):
            for l in range(n + 1):
                d1 = m - k + l
                d2 = n + k - l
                p[d1, d2, m, n, k, l] = 1

p1 = tf.constant(p, tf.complex128)


r_pow = np.zeros((L,)*4, dtype=int)
t_pow = np.zeros((L,)*4, dtype=int)
for m in range(L):
    for n in range(L):
        for k in range(m + 1):
            for l in range(n + 1):
                r_pow[m, n, k, l] = k + l
                t_pow[m, n, k, l] = m - k + n - l

r_pows_tf = tf.constant(r_pow, tf.float64)
t_pows_tf = tf.constant(t_pow, tf.float64)

t1_tf = tf.pow(tf.sqrt(T1), t_pows_tf)
r1_tf = tf.pow(tf.sqrt(tf.add(tf.negative(T1), 1)), r_pows_tf)


m1 = tf.tensordot(mut_state, tf.ones([L, L], tf.complex128), axes=0)
m2 = tf.multiply(tf.cast(tf.multiply(t1_tf, r1_tf), tf.complex128), bs_const_tf)

c1 = tf.multiply(m1, m2)

c2 = tf.tensordot(tf.ones([L*2 - 1, L*2 - 1], tf.complex128), c1, axes=0)

d = tf.multiply(p1, c2)

state_out = tf.reduce_sum(d, axis=[2, 3, 4, 5])


with tf.Session() as sess:
    st_out = state_out.eval()
