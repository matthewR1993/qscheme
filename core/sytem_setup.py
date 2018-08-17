from .basic import *


# Process the whole system.
# Input:
#  input_state - input unapplied state in 2 channels
#  bs_params - BS parameters as a dict.
#  phase_diff - phase modulation in [rad]
#  det_event - detection option
# Output:
#  applied density matrix in 2 channels
def process_all(input_state, bs_params, phase_diff, det_event):
    t1, r1 = bs_params['t1'], bs_params['r1']
    t2, r2 = bs_params['t2'], bs_params['r2']
    t3, r3 = bs_params['t3'], bs_params['r3']
    t4, r4 = bs_params['t4'], bs_params['r4']

    # First BS.
    state_after_bs_unappl = bs2x2_transform(t1, r1, input_state)

    # 2d and 3rd BS.
    state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)

    # Detection probability
    det_prob = det_probability(state_aft2bs_unappl, detection_event=det_event)
    # print('det. prob.', det_prob)

    # The detection event.
    # Gives non-normalised state.
    state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=det_event)

    # Calculating the norm.
    norm_after_det = state_norm(state_after_dett_unappl)
    # print('Norm after det.:', norm_after_det)

    # The normalised state.
    state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det

    # trim the state
    trim_state = 8
    state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state, :trim_state]
    sm_state = np.sum(np.abs(state_after_dett_unappl_norm[trim_state:, trim_state:, trim_state:, trim_state:]))
    if sm_state > 1e-10:
        print('State trim norm:', sm_state)

    # Building dens. matrix and trace.
    dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)

    # Phase modulation
    dens_matrix_2channels_withph = phase_modulation(dens_matrix_2channels, phase_diff)

    # The transformation at last BS
    trim_dm = 6
    final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels_withph[:trim_dm, :trim_dm, :trim_dm, :trim_dm], t4, r4)
    sm_dm = np.sum(np.abs(dens_matrix_2channels_withph[trim_dm:, trim_dm:, trim_dm:, trim_dm:]))
    if sm_dm > 1e-10:
        print('Dens. matr. trim norm:', sm_dm)

    return final_dens_matrix, det_prob
