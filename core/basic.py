import cmath
import numpy as np
from math import sqrt, factorial


# returns 2x2 BS transformation matrix
def bs2x2_transform(t, r, input_state):
    size = len(input_state)
    output_state = np.zeros((size*2 - 1, size*2 - 1), dtype=complex)

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
                    coeff = input_state[m, n] * t1**(m - k) * (1j*r1)**k * t2**(n - l) * (1j*r2)**l * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))
                    output_state[ind1, ind2, ind3, ind4] = output_state[ind1, ind2, ind3, ind4] + coeff

    return output_state


# simple solution, 4 channels state
def detection(input_state, detection_event='FIRST'):
    size = len(input_state)
    output_state = np.array(input_state)
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


def state_norm(state):
    # takes unapplied state
    size = len(state)
    norm_ = 0
    for p1 in range(size):
        for p2 in range(size):
            for p3 in range(size):
                for p4 in range(size):
                    norm_ = norm_ + abs(state[p1, p2, p3, p4])**2 * factorial(p1)*factorial(p2)*factorial(p3)*factorial(p4)
    return sqrt(norm_)


def dens_matrix_with_trace(left_vector, right_vector):
    size = len(left_vector)
    if len(left_vector) != len(right_vector):
        raise ValueError('Incorrect dimensions')

    right_vector_conj = np.conj(right_vector)
    dens_matrix = np.zeros((size,) * 4, dtype=complex)

    for p2 in range(size):
        for p2_ in range(size):
            for p4 in range(size):
                for p4_ in range(size):
                    matrix_sum = 0
                    for k1 in range(size):
                        for k3 in range(size):
                            matrix_sum = matrix_sum + left_vector[k1, p2, k3, p4] * right_vector_conj[k1, p2_, k3, p4_] * factorial(k1) * factorial(k3) * sqrt(factorial(p2)*factorial(p4)*factorial(p2_)*factorial(p4_))
                    dens_matrix[p2, p4, p2_, p4_] = matrix_sum
    return dens_matrix


# trace one channel
def trace_channel(input_matrix, channel=4):
    size = len(input_matrix)
    reduced_matrix = np.zeros((size, size), dtype=complex)
    if channel is 4:
        for p2 in range(size):
            for p2_ in range(size):
                sum = 0
                for n in range(size):
                    sum = sum + input_matrix[p2, n, p2_, n]
                reduced_matrix[p2, p2_] = sum
    elif channel is 2:
        for p4 in range(size):
            for p4_ in range(size):
                sum = 0
                for n in range(size):
                    sum = sum + input_matrix[n, p4, n, p4_]
                reduced_matrix[p4, p4_] = sum
    else:
        raise ValueError('Invalid configuration')
    return reduced_matrix


# Last beam splitter transformation of dens matrix.
def bs_densmatrix_transform(input_matrix, t4, r4):
    size = len(input_matrix)
    output_matrix = np.zeros((size*2,) * 4, dtype=complex)

    for p2 in range(size):
        for p4 in range(size):
            for p2_ in range(size):
                for p4_ in range(size):

                    # two sums up to m and n
                    for k in range(p2 + 1):
                        for l in range(p4 + 1):
                            for k_ in range(p2_ + 1):
                                for l_ in range(p4_ + 1):
                                    d1 = p2 - k + l
                                    d2 = p4 - l + k
                                    coeff1 = t4**(p2 - k + p4 - l) * (1j*r4)**(l+k) * sqrt(factorial(d1)*factorial(d2)) * factorial(p2)*factorial(p4)/(factorial(k)*factorial(p2-k)*factorial(l)*factorial(p4-l))

                                    d1_ = p2_ - k_ + l_
                                    d2_ = k_ + p4_ - l_
                                    coeff2 = t4**(p2_ - k_ + p4_ - l_) * (-1j*r4)**(k_ + l_) * sqrt(factorial(d1_)*factorial(d2_)) * factorial(p2_)*factorial(p4_)/(factorial(k_)*factorial(p2_-k_)*factorial(l_)*factorial(p4_-l_))

                                    output_matrix[d1, d2, d1_, d2_] = output_matrix[d1, d2, d1_, d2_] + input_matrix[p2, p4, p2_, p4_] * 1/(sqrt(factorial(p2)*factorial(p4)*factorial(p2_)*factorial(p4_))) * coeff1 * coeff2

    return output_matrix


# photons distribution probability from final_dens_matrix
def prob_distr(input_matrix):
    size = len(input_matrix)
    prob_matrix = np.zeros((size, size), dtype=complex)
    for m in range(size):
        for n in range(size):
            prob_matrix[m, n] = input_matrix[m, n, m, n]

    return prob_matrix


# The logarithmic entanglement out of 2x2 density matrix
def log_entropy(dens_matrix):
    size = len(dens_matrix)
    entropy = 0
    w, v = np.linalg.eig(dens_matrix)
    for n in range(size):
        if w[n] != 0:
            entropy = entropy + w[n] * np.log(w[n])
    entropy = - entropy
    return entropy


# partial transpose of 2 channels dens. matrix
def partial_transpose(matrix):
    size = len(matrix)
    res_matrix = np.zeros((size,) * 4, dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):
                    res_matrix[p1, p2, p1_, p2_] = matrix[p1, p2_, p1_, p2]
    return res_matrix


def linear_entropy(dens_matrix):
    entropy = 1 - np.trace(dens_matrix @ dens_matrix)
    return entropy


def reorganise_dens_matrix(rho):
    size = len(rho)
    rho_out = np.zeros((size**2,)*2, dtype=complex)
    for m in range(size):
        for n in range(size):
            for m_ in range(size):
                for n_ in range(size):
                    k = m * size + n
                    k_ = m_ * size + n_
                    rho_out[k, k_] = rho[m, n, m_, n_]
    return rho_out


def negativity(rho, neg_type='logarithmic'):
    part_transposed = partial_transpose(rho)
    reorg_rho = reorganise_dens_matrix(part_transposed)
    w, v = np.linalg.eig(reorg_rho)
    neg = 0
    for eigval in w:
        if np.real(eigval) < 0:
            neg = neg + np.abs(np.real(eigval))
    if neg_type is 'logarithmic':
        return np.log2(2 * neg + 1)
    elif neg_type is 'raw':
        return neg
    else:
        raise ValueError('Incorrect configuration')
