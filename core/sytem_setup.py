from .basic import *
from core.optimized import transformations as trans
import time


def process_all(input_state, bs_params, phase_diff, phase_mod_channel, det_event):
    """
    Process the whole system.
    :param input_state: input unapplied state in 2 channels
    :param bs_params: BS parameters as a dict.
    :param phase_diff: phase modulation in [rad]
    :param phase_mod_channel: number of channel for the phase modulation
    :param det_event: detection option
    :return: applied density matrix in 2 channels
    """
    t1, r1 = bs_params['t1'], sqrt(1 - bs_params['t1']**2)
    t2, r2 = bs_params['t2'], sqrt(1 - bs_params['t2']**2)
    t3, r3 = bs_params['t3'], sqrt(1 - bs_params['t3']**2)
    t4, r4 = bs_params['t4'], sqrt(1 - bs_params['t4']**2)

    # First BS.
    state_after_bs_unappl = bs2x2_transform(t1, r1, input_state)

    # 2d and 3rd BS.
    # state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)
    state_aft2bs_unappl = trans.two_bs2x4_transform_copt(t2, r2, t3, r3, state_after_bs_unappl)

    # Detection probability
    det_prob = det_probability(state_aft2bs_unappl, detection_event=det_event)
    # print('det. prob.', det_prob)

    # The detection event.
    # Gives non-normalised state.
    state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=det_event)

    # Calculating the norm.
    norm_after_det = state_norm_opt(state_after_dett_unappl)
    # print('Norm after det.:', norm_after_det)

    # The normalised state.
    state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det
    if norm_after_det < 1e-12:
        print('Norm after detection is too low.')
        return (None,) * 3

    # Trim the state, 8 is min.
    trim_state = 8  # 8
    state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state, :trim_state]
    # sm_state = np.sum(np.abs(state_after_dett_unappl_norm)) - np.sum(np.abs(state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state, :trim_state]))
    # print('State trim norm:', sm_state)

    # Building dens. matrix and trace.
    dens_matrix_2ch = dens_matrix_with_trace_opt(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)

    # Phase modulation
    dens_matrix_2channels_withph = phase_modulation(dens_matrix_2ch, phase_diff, channel=phase_mod_channel)

    # The transformation at last BS, 7 is min.
    trim_dm = 7  # 7
    final_dens_matrix = trans.bs_matrix_transform_copt(dens_matrix_2channels_withph[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t4, r4)
    # sm_dm = np.sum(np.abs(dens_matrix_2channels_withph)) - np.sum(np.abs(dens_matrix_2channels_withph[:trim_dm, :trim_dm, :trim_dm, :trim_dm]))
    # print('Dens. matr. trim norm:', sm_dm)

    # sum = 0
    # for n in range(len(final_dens_matrix)):
    #     sum += final_dens_matrix[n, n, n, n]
    # print("Sum:", sum)

    return final_dens_matrix, det_prob, norm_after_det
