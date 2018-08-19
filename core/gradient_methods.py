from core.sytem_setup import *
from core.squeezing import *


# Maximum number of iterations for a gradient descent.
SEARCH_ITER_MAX = 40

PAR_KEYS = ['t1', 't2', 't3', 't4']


def gradient_descent(start_point, state, phase_diff, det_event, quantity=None, delta_t=1e-6, prec_t=1e-4):
    bs_params_arr = np.zeros(SEARCH_ITER_MAX + 1, dtype=dict)
    bs_params_arr[0] = start_point

    # Calculating gradients.
    for i in range(SEARCH_ITER_MAX):
        dm, _ = process_all(state, bs_params_arr[i], phase_diff=phase_diff, det_event=det_event)
        if quantity in ['EPR_X', 'EPR_P']:
            epr_x, epr_p = erp_squeezing_correlations(dm)
        elif quantity in ['QUADR_X', 'QUADR_P']:
            dX, dP = squeezing_quadratures(dm, channel=1)

        grads = np.zeros(len(PAR_KEYS), dtype=complex)
        for j in range(len(PAR_KEYS)):
            bs_params_upp = bs_params_arr[i].copy()
            bs_params_upp[PAR_KEYS[j]] += delta_t
            dm_up, _ = process_all(state, bs_params_upp, phase_diff=phase_diff, det_event=det_event)
            if quantity is 'EPR_X':
                epr_x_up, _ = erp_squeezing_correlations(dm_up)
                grads[j] = (epr_x_up - epr_x) / delta_t
                ret_val = epr_x
            elif quantity is 'EPR_P':
                _, epr_p_up = erp_squeezing_correlations(dm_up)
                grads[j] = (epr_p_up - epr_p) / delta_t
                ret_val = epr_p
            elif quantity is 'QUADR_X':
                dX_up, _ = squeezing_quadratures(dm_up, channel=1)
                grads[j] = (dX_up - dX) / delta_t
                ret_val = dX
            elif quantity is 'QUADR_X':
                _, dP_up = squeezing_quadratures(dm_up, channel=1)
                grads[j] = (dP_up - dP) / delta_t
                ret_val = dP
            else:
                raise ValueError

        print(grads)
        # Build up the next step by adding gradient.
        bs_params_next = bs_params_arr[i].copy()
        for j in range(len(PAR_KEYS)):
            bs_params_next[PAR_KEYS[j]] -= delta_t * grads[j]
        bs_params_arr[i + 1] = bs_params_next

        print(bs_params_next)

        print('t diff:', (np.abs(grads)) * delta_t)
        if np.max(np.abs(grads)) * delta_t <= prec_t:
            return {
                'step': i,
                'min_par': bs_params_arr[i],
                'min_val': ret_val
            }

    return None
