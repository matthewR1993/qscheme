import cmath
import numpy as np
from scipy.misc import factorial as fact
from math import sqrt, factorial


def bs2x2_transform(t, r, input_state):
    '''
    Two channels (2x2) beam splitter transformation.
    With: t^2 + r^2 = 1.
    :param t: Transmission coefficient.
    :param r: Reflection coefficient.
    :param input_state: Unapplied state in two channels(modes).
    :return: Transformed unapplied state in two channels(modes).
    '''
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


def two_bs2x4_transform(t1, r1, t2, r2, input_state):
    '''
    Transformation at 2 beam splitters.
    Two input channels and four output channles - 2x4 transformation.
    Creation operators transformation:
    a1 => t1 a2 + i r1 a1.
    a2 => t2 a4 + i r2 a3.
    With transmission and reflection coefficients:
    t1^2 + r1^2 = 1.
    t2^2 + r2^2 = 1.
    :param t1: BS1 transmission.
    :param r1: BS1 reflection.
    :param t2: BS2 transmission.
    :param r2: BS2 reflection.
    :param input_state: Two channels(modes) unapllied state.
    :return: Four channels(modes) unapllied state.
    '''
    size = len(input_state)
    output_state = np.zeros((size,) * 4, dtype=complex)
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


def detection(input_state, detection_event):
    '''
    Tranformation of the state with POVM operator.
    :param input_state: Applied/unapplied state in 4 channels(modes).
    :param detection_event: Detection event.
    :return: Applied/unapplied state in 4 channels(modes).
    '''
    size = len(input_state)
    output_state = np.zeros((size,) * 4, dtype=complex)
    if detection_event is 'BOTH':
        for p1 in range(size):
            for p2 in range(size):
                for p3 in range(size):
                    for p4 in range(size):
                        if p1 is not 0 and p3 is not 0:
                            output_state[p1, p2, p3, p4] = input_state[p1, p2, p3, p4]
    elif detection_event is 'NONE':
        for p1 in range(size):
            for p2 in range(size):
                for p3 in range(size):
                    for p4 in range(size):
                        if p1 is 0 and p3 is 0:
                            output_state[p1, p2, p3, p4] = input_state[p1, p2, p3, p4]
    elif detection_event is 'FIRST':
        for p1 in range(size):
            for p2 in range(size):
                for p3 in range(size):
                    for p4 in range(size):
                        if p1 > 0 and p3 is 0:
                            output_state[p1, p2, p3, p4] = input_state[p1, p2, p3, p4]
    elif detection_event is 'THIRD':
        for p1 in range(size):
            for p2 in range(size):
                for p3 in range(size):
                    for p4 in range(size):
                        if p1 is 0 and p3 is not 0:
                            output_state[p1, p2, p3, p4] = input_state[p1, p2, p3, p4]
    else:
        raise ValueError('Wrong configuration')

    return output_state


def det_probability(input_state, detection_event):
    '''
    Calculating a probability of an event realisation.
    :param input_state: Unapplied state in 4 channels.
    :param detection_event: Detection event.
    :return: Probability of the detection.
    '''
    size = len(input_state)
    st_aft_det_unappl = detection(input_state, detection_event)
    st_aft_det_unappl_conj = np.conj(st_aft_det_unappl)
    trace = 0
    for p1 in range(size):
        for p2 in range(size):
            for p3 in range(size):
                for p4 in range(size):
                    trace = trace + input_state[p1, p2, p3, p4] * st_aft_det_unappl_conj[p1, p2, p3, p4]
    return 1 - trace


def state_norm(state):
    '''
    A norm of the state.
    :param state: Unapplied state in 4 channels.
    :return: Norm of the state.
    '''
    size = len(state)
    norm_ = 0
    for p1 in range(size):
        for p2 in range(size):
            for p3 in range(size):
                for p4 in range(size):
                    norm_ = norm_ + abs(state[p1, p2, p3, p4])**2 * factorial(p1)*factorial(p2)*factorial(p3)*factorial(p4)
    return sqrt(norm_)


def dens_matrix_with_trace(left_vector, right_vector):
    '''
    Composing density matrix from projected vectors and partially trace.
    :param left_vector: Ket unapplied state in 4 channels.
    :param right_vector: Bra unapplied state in 4 channels.
    :return: Applied dens matrix for 2 channels.
    '''
    size = len(left_vector)
    if len(left_vector) != len(right_vector):
        raise ValueError('Incorrect dimensions')

    right_vector_conj = np.conj(right_vector)
    dm = np.zeros((size,) * 4, dtype=complex)

    for p2 in range(size):
        for p2_ in range(size):
            for p4 in range(size):
                for p4_ in range(size):
                    matrix_sum = 0
                    for k1 in range(size):
                        for k3 in range(size):
                            matrix_sum = matrix_sum + left_vector[k1, p2, k3, p4] * right_vector_conj[k1, p2_, k3, p4_] * factorial(k1) * factorial(k3) * sqrt(factorial(p2)*factorial(p4)*factorial(p2_)*factorial(p4_))
                    dm[p2, p4, p2_, p4_] = matrix_sum

    return dm


def dens_matrix(state):
    '''
    Build a density matrix in 2 channels.
    :param state: Applied state in 2 channels.
    :return: Applied density matrix for 2 channels.
    '''
    size = len(state)
    state_conj = np.conj(state)
    dm = np.zeros((size,) * 4, dtype=complex)

    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):
                    dm[p1, p2, p1_, p2_] = state[p1, p2] * state_conj[p1_, p2_]

    return dm


def dens_matrix_4ch(state):
    '''
    Build a density matrix in 4 channels.
    :param state: Applied state in 4 channels.
    :return: Applied density matrix for 4 channels.
    '''
    size = len(state)
    state_conj = np.conj(state)
    dens_matrix = np.zeros((size,) * 8, dtype=complex)

    for p1 in range(size):
        for p2 in range(size):
            for p3 in range(size):
                for p4 in range(size):
                    for p1_ in range(size):
                        for p2_ in range(size):
                            for p3_ in range(size):
                                for p4_ in range(size):
                                    dens_matrix[p1, p2, p3, p4, p1_, p2_, p3_, p4_] = state[p1, p2, p3, p4] * state_conj[p1_, p2_, p3_, p4_]

    return dens_matrix


def trace_channel(input_matrix, channel=4):
    '''
    Tracing one channel of density matrix in 2 channels(modes).
    :param input_matrix: Applied density matrix in 2 channels.
    :param channel: Number of the channel.
    :return: Applied reduced density matrix of one channel.
    '''
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


def bs_densmatrix_transform(input_matrix, t, r):
    '''
    Beam splitter transformation of density matrix in 2 channels.
    Mapping of creation operators:
    a2 => t b1 + i r b2.
    a4 => t b2 + i r b1.
    :param input_matrix: Applied density matrix in 2 channels.
    :param t: Transmission coefficient.
    :param r: Reflection coefficient.
    :return: Applied density matrix in 2 channels.
    '''
    size = len(input_matrix)
    output_matrix = np.zeros((size*2,) * 4, dtype=complex)

    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):

                    # four sums
                    for n in range(p1 + 1):
                        for k in range(p2 + 1):
                            for n_ in range(p1_ + 1):
                                for k_ in range(p2_ + 1):
                                    d1 = p1 - n + k
                                    d2 = n + p2 - k
                                    coeff1 = t**(p1 - n + p2 - k) * (1j*r)**(n + k) * sqrt(factorial(d1) * factorial(d2) * factorial(p1) * factorial(p2)) / (factorial(n) * factorial(p1 - n) * factorial(k) * factorial(p2 - k))

                                    d1_ = p1_ - n_ + k_
                                    d2_ = n_ + p2_ - k_
                                    coeff2 = t**(p1_ - n_ + p2_ - k_) * (-1j*r)**(n_ + k_) * sqrt(factorial(d1_) * factorial(d2_) * factorial(p1_) * factorial(p2_)) / (factorial(n_) * factorial(p1_ - n_) * factorial(k_) * factorial(p2_ - k_))

                                    output_matrix[d1, d2, d1_, d2_] = output_matrix[d1, d2, d1_, d2_] + input_matrix[p1, p2, p1_, p2_] * coeff1 * coeff2

    return output_matrix


def prob_distr(input_matrix):
    '''
    Photons distribution probability from final density matrix.
    :param input_matrix: Applied density matrix in 2 channels.
    :return: Probability distribution for 2 channels.
    '''
    size = len(input_matrix)
    prob_matrix = np.zeros((size, size), dtype=complex)
    for m in range(size):
        for n in range(size):
            prob_matrix[m, n] = input_matrix[m, n, m, n]

    return prob_matrix


def log_entropy(dm):
    '''
    Calculating logarithmic Fon Neuman entropy / entanglement.
    :param dm: Applied reduced density matrix.
    :return: Entropy.
    '''
    size = len(dm)
    entropy = 0
    w, v = np.linalg.eig(dm)
    for n in range(size):
        if w[n] != 0:
            entropy = entropy - w[n] * np.log2(w[n])
    return entropy


def partial_transpose(matrix):
    '''
    Partial transpose of 2 channels density matrix.
    :param matrix: Density matrix in 2 channels.
    :return: Density matrix in 2 channels.
    '''
    size = len(matrix)
    res_matrix = np.zeros((size,) * 4, dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):
                    res_matrix[p1, p2, p1_, p2_] = matrix[p1, p2_, p1_, p2]
    return res_matrix


def linear_entropy(dm):
    '''
    Linear entropy.
    :param dm: Reduced density matrix.
    :return: Linear entropy.
    '''
    entropy = 1 - np.trace(dm @ dm)
    return entropy


def reorganise_dens_matrix(rho):
    '''
    Reorganise density matrix in 2 channels:
    rho[m, n, m_, n_] ==> rho_out[k, k_].
    :param rho: Density matrix in 2 channels.
    :return: Reorganised density matrix in two channels.
    '''
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
    '''
    Calculating negativity for 2 channels.
    :param rho: Applied density matrix in 2 channels.
    :param neg_type: Negativity type.
    :return: Negativity.
    '''
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


def phase_modulation(rho, phase):
    '''
    A phase modulation for the density matrix in 2 channels.
    :param rho: Density matrix in 2 channels.
    :param phase: Phase.
    :return: Modulated density matrix in 2 channels.
    '''
    if phase is 0:
        return rho
    size = len(rho)
    rho_out = np.zeros((size,)*4, dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):
                    rho_out[p1, p2, p1_, p2_] = rho[p1, p2, p1_, p2_] * np.exp(1j * phase * (p2 - p2_))
    return rho_out


def phase_modulation_state(state, phase):
    '''
    A phase modulation for the state in two channels.
    :param state: Unapplied state in 2 channels.
    :param phase: Phase.
    :return: Modulated unapplied state in 2 channels.
    '''
    size = len(state)
    st_mod = np.zeros((size, size), dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            st_mod[p1, p2] = state[p1, p2] * np.exp(1j * p1 * phase)
    return st_mod


def make_state_appliable(state):
    '''
    Apply operators to the state in 2 channels.
    :param state: Unapplied state in 2 channels.
    :return: Applied state in 2 channels.
    '''
    size = len(state)
    st_appl = np.zeros((size, size), dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            st_appl[p1, p2] = state[p1, p2] * sqrt(factorial(p1) * factorial(p2))
    return st_appl


def make_state_appliable_4ch(state):
    '''
    Apply operators to state in 4 channels.
    :param state: Unapplied state in 4 channels.
    :return: Applied state in 4 channels.
    '''
    size = len(state)
    st_appl = np.zeros((size,)*4, dtype=complex)
    for p1 in range(size):
        for p2 in range(size):
            for p3 in range(size):
                for p4 in range(size):
                    st_appl[p1, p2, p3, p4] = state[p1, p2, p3, p4] * sqrt(factorial(p1) * factorial(p2) * factorial(p3) * factorial(p4))
    return st_appl


def bs_params(T_min, T_max, num):
    '''
    Generating BS's t and r parameters arrays.
    :param T_min: T min.
    :param T_max: T max.
    :param num: length.
    :return: BS's t and r small coeficients.
    '''
    T_array = np.linspace(T_min, T_max, num)
    t_array = np.sqrt(T_array)
    rf = np.vectorize(lambda t: sqrt(1 - pow(t, 2)))
    r_array = rf(t_array)
    return t_array, r_array
