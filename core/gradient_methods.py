from core.sytem_setup import *
from core.squeezing import *


# Maximum number of iterations for a gradient descent..
SEARCH_ITER_MAX = 40

PAR_KEYS = ['t1', 't2', 't3', 't4']


# TODO
def gradient_descent(quantity, start_point, state, phase_diff, det_event, delta=1e-7):
    bs_params = {
        't1': start_point[0],
        'r1': sqrt(1 - start_point[0]**2),
        't4': start_point[3],
        'r4': sqrt(1 - start_point[3]**2),
        't2': start_point[1],
        'r2': sqrt(1 - start_point[1]**2),
        't3': start_point[2],
        'r3': sqrt(1 - start_point[2]**2),
    }

    # Start point.
    bs_params_arr = [bs_params]
    dm, _ = process_all(state, bs_params_arr[0], phase_diff=phase_diff, det_event=det_event)

    # Calculating gradients.
    for i in range(1, SEARCH_ITER_MAX):
        dm, _ = process_all(state, bs_params_arr[i], phase_diff=phase_diff, det_event=det_event)
        if quantity in ['epr_x', 'epr_p']:
            epr_x, epr_p = erp_squeezing_correlations(dm)
        elif quantity in ['quadr_x', 'quadr_p']:
            dX, dP = squeezing_quadratures(dm, channel=1)

        grads = np.zeros(len(PAR_KEYS), dtype=complex)
        for j in range(len(PAR_KEYS)):
            bs_params_upp = bs_params_arr[i].copy()
            bs_params_upp[PAR_KEYS[j]] += delta
            dm_up, _ = process_all(state, bs_params_upp, phase_diff=phase_diff, det_event=det_event)
            if quantity is 'epr_x':
                epr_x_up, _ = erp_squeezing_correlations(dm_up)
                grads[j] = (epr_x_up - epr_x) / delta
            elif quantity is 'epr_p':
                _, epr_p_up = erp_squeezing_correlations(dm_up)
                grads[j] = (epr_p_up - epr_p) / delta
            elif quantity is 'quadr_x':
                dX_up, _ = squeezing_quadratures(dm_up, channel=1)
                grads[j] = (dX_up - dX) / delta
            elif quantity is 'quadr_p':
                _, dP_up = squeezing_quadratures(dm_up, channel=1)
                grads[j] = (dP_up - dP) / delta

        # build next step summing up gradient.
        bs_params_next = bs_params_arr[i].copy()
        bs_params_next['t1'] += 0
        bs_params_arr[i + 1] = bs_params_next


    return 0
