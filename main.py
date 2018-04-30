import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

from customutils.utils import *
from core.projection import measure_state
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *
import tensorflow as tf
from qutip.operators import create

sess = tf.Session()

# Parameters for states
series_length = 12
input_series_length = 12
auxiliary_series_length = 12
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!

# INPUT
input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
#print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
# auxiliary_st = single_photon(2)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement detectors configuration
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector clicked
# DET_CONF = 'NONE'  # None of detectors was clicked

diagonal_factorials = np.identity(input_series_length) * np.array([sqrt(factorial(x)) for x in range(input_series_length)])

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

diagonal_factorials_tf = tf.constant(diagonal_factorials, tf.float64)

in_state_tf_appl = tf.einsum('mn,n->n', diagonal_factorials_tf, in_state_tf)
aux_state_tf_appl = tf.einsum('mn,n->n', diagonal_factorials_tf, aux_state_tf)


# print it
# diagonal_factorials_tf.eval(session=sess)

# simple multiplication
tens_mult = (in_state_tf * aux_state_tf).eval(session=sess)

# tensor product, return numpy array
prod_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

plt.matshow(np.abs(prod_unappl))
plt.colorbar()
plt.show()


#ccc = tf.multiply(
#    in_state_tf,
#    in_state_tf,
#    name=None
#).eval(session=sess)


# Setting up state before first BS.
a1, a2 = sp.symbols('a1 a2')
g = 0
for i in range(len(input_st)):
    g = g + input_st[i]*(a1**i)
f = 0
for i in range(len(auxiliary_st)):
    f = f + auxiliary_st[i]*(a2**i)

# g(a1) - input
# f(a2) - auxiliary
# Initial state = g(a1) * f(a2)
state1 = g * f

state1_coeffs_unapp = get_state_coeffs(sp.expand(state1), max_power + 1, operators_form='unapplied')

plot_state(state1_coeffs_unapp, 'Initial State',  size=10, value='real')


# returns 2x2 BS transformation matrix
def bs2x2_transform(t, r, input_state):
    size = len(input_state)
    output_state = np.zeros((size*2, size*2), dtype=complex)

    for m in range(size):
        for n in range(size):
            # two sums up to m and n
            for k in range(m + 1):
                for l in range(n + 1):
                    first_index = m - k + l  # first channel index
                    second_index = k + n - l  # second channel index
                    coeff = input_state[m, n] * (1j*r)**(k + l) * t**(m - k + n - l) * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))
                    output_state[first_index, second_index] = output_state[first_index, second_index] + coeff

    return output_state

# better
state_after_bs_unappl = bs2x2_transform(t1, r1, prod_unappl)

plot_state(state_after_bs_unappl, 'Initial State',  size=10, value='abs')


# State after mixing at first BS
#state2 = state1
#b1, b2 = sp.symbols('b1 b2')

# a1 -> t1*a1 + 1j*r1*a2
#state2 = state2.subs(a1, (t1*b1 + 1j*r1*b2))
#state2 = state2.subs(a2, (t1*b2 + 1j*r1*b1))

# put 'a' operators back
#state2 = state2.subs(b1, a1)
#state2 = state2.subs(b2, a2)

#state2 = sp.expand(state2)
#print('State 2:', state2)

# Plot state2
#state2_coeffs = get_state_coeffs(state2, max_power + 1, operators_form='unapplied')

#plot_state(state2_coeffs, 'State2',  size=8, value='abs')
# plot_state(state2_coeffs, 'State2',  size=8, value='real')
# plot_state(state2_coeffs, 'State2',  size=8, value='imag')

# 'state2' is a state after BS

# a1 goes to 2nd BS with t2, r2 and split into b1 and b2. Therefore: a1 -> t2*b1 + 1j*r2*b2
# a2 goes to 3rd BS with t3, r3 and split into b3 and b4. Therefore: a2 -> t3*b3 + 1j*r3*b4
# state3 is a state after these two BSs
# state3 = state2
# b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')

# state3 = state3.subs(a1, (t2*b2 + 1j*r2*b1))
# state3 = state3.subs(a2, (t3*b4 + 1j*r3*b3))

# state3 = sp.expand(state3)

# print('State 3:', state3)


# 2 channels : 2 BS : 4 channels
def two_bs2x4_transform(t1, r1, t2, r2, input_state):
    size = len(input_state)
    output_state = np.zeros((size, size, size, size), dtype=complex)
    for m in range(size):
        for n in range(size):
            # two sums up to m and n
            for k in range(m + 1):
                for l in range(n + 1):
                    # channels indexes
                    ind1 = k
                    ind2 = m - k
                    ind3 = l
                    ind4 = n - l
                    coeff = input_state[m, n] * t1**(m - k) * (1j*r1)**k * t2**(n - l) * 1j*r2**l * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))
                    output_state[ind1, ind2, ind3, ind4] = output_state[ind1, ind2, ind3, ind4] + coeff

    return output_state


state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)


# simple solution, 4 channels state
def detection(input_state, detection_event='FIRST'):
    size = len(input_state)
    output_state = input_state
    if detection_event is 'BOTH':
        pass
    elif detection_event is 'NONE':
        output_state[0, :, :, :] = 0
        output_state[:, :, 0, :] = 0
    elif detection_event is 'FIRST':
        output_state[0, :, :, :] = 0
        for p1 in range(size):
            for p2 in range(size):
                for p3 in range(size):
                    for p4 in range(size):
                        if p3 > 0:
                            output_state[p1, p2, p3, p4] = 0
    elif detection_event is 'THIRD':
        output_state[:, :, 0, :] = 0
        for p1 in range(size):
            for p2 in range(size):
                for p3 in range(size):
                    for p4 in range(size):
                        if p1 > 0:
                            output_state[p1, p2, p3, p4] = 0
    else:
        raise ValueError('Wrong configuration')

    return output_state


state_after_dett = detection(state_aft2bs_unappl, detection_event='FIRST')

# Build  8D density matrix
state_after_dett_tf = tf.constant(state_after_dett, tf.complex128)
state_before_dett_tf = tf.constant(state_aft2bs_unappl, tf.complex128)


def build_dens_matrix(left_vector, right_vector):
    size = len(left_vector)
    if len(left_vector) != len(right_vector):
        raise ValueError('Incorrect dimensions')

    dens_matrix = np.zeros((size,)*8, dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            for p3 in range(size):
                for p4 in range(size):
                    for p1_ in range(size):
                        for p2_ in range(size):
                            for p3_ in range(size):
                                for p4_ in range(size):
                                    dens_matrix[p1, p2, p3, p4, p1_, p2_, p3_, p4_] = left_vector[p1, p2, p3, p4] * np.conjugate(right_vector[p1_, p2_, p3_, p4_])
    return dens_matrix


dens_matr_aft_det = build_dens_matrix(state_after_dett, state_after_dett)


# TODO
def partial_trace():
    pass


# state4 is a state after measurement
#state_4pre = 0

#state4 = measure_state(state3, clicked=DET_CONF)

#print('State 4:', state4)

# State 4 plots
# state4_coeffs = get_state_coeffs(state4.subs(b2, a1).subs(b4, a2), max_power)
# plot_state(state4_coeffs, 'State4',  size=8, value='abs', xlabel='b4 degree', ylabel='b2 degree')
# plot_state(state4_coeffs, 'State4',  size=8, value='real', xlabel='b4 degree', ylabel='b2 degree')
# plot_state(state4_coeffs, 'State4',  size=8, value='imag', xlabel='b4 degree', ylabel='b2 degree')


# Now mixing state in a fourth BS
# Final state is state5
# b2 -> t4*a1 + 1j*t4*a2
# b4 -> t4*a2 + 1j*t4*a1
state5 = state4

state5 = state5.subs(b2, (t4*a1 + 1j*r4*a2))
state5 = state5.subs(b4, (t4*a2 + 1j*r4*a1))

state5 = sp.expand(state5)

print('State 5:', state5)

# Plotting final state.
# Matrix of coefficients.
state5_coeffs = get_state_coeffs(state5, max_power)

plot_state(state5_coeffs, 'Final State',  size=8, value='abs')
plot_state(state5_coeffs, 'Final State',  size=8, value='real')
plot_state(state5_coeffs, 'Final State',  size=8, value='imag')

'''

# plot input states
plt.bar(list(range(len(input_st))), input_st, width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], width=1, edgecolor='c')
plt.title('Input state')
plt.xlabel('Number of photons')
plt.show()
plt.bar(list(range(len(auxiliary_st))), auxiliary_st, color='g', width=1, edgecolor='c')
# plt.bar(list(range(8)), [0, 1, 0, 0, 0, 0, 0, 0], color='g', width=1, edgecolor='c')
plt.title('Auxiliary state')
plt.xlabel('Number of photons')
plt.show()

'''

# save setup configuration in file TODO
