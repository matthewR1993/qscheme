from core.sytem_setup import *
from core.squeezing import *


def gradient_descent(state, phase_diff, det_event, params, quantity=None):
    '''
    A gradient descent method with the momentum.
    :param state: Input state.
    :param phase_diff: Phase.
    :param det_event: Detection event.
    :param params: Algorithm parameters.
    :param quantity: Minimized quantity.
    :return: Object.
    '''
    delta_t = params['alpha']
    prec_t = params['target_prec']
    betta = params['betta']
    start_point = params['start_point']
    par_keys = params['par_keys']  # ['t1', 't2', 't3', 't4']
    fixed_params = params['fixed_parameters']  # [{'t2': 0.5}, {'t3': 0.5}]
    search_iter_max = params['search_iter_max']  # 40

    bs_params_arr = np.zeros(search_iter_max + 1, dtype=dict)
    bs_params_arr[0] = start_point

    # Calculating gradients.
    for i in range(search_iter_max):
        dm, _ = process_all(state, bs_params_arr[i], phase_diff=phase_diff, det_event=det_event)
        if quantity in ['EPR_X', 'EPR_P']:
            epr_x, epr_p = erp_squeezing_correlations(dm)
        elif quantity in ['QUADR_X', 'QUADR_P']:
            dX, dP = squeezing_quadratures(dm, channel=1)

        grads = np.zeros(len(par_keys), dtype=complex)
        for j in range(len(par_keys)):
            bs_params_upp = bs_params_arr[i].copy()
            bs_params_upp[par_keys[j]] += delta_t
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
        for j in range(len(par_keys)):
            bs_params_next[par_keys[j]] -= delta_t * grads[j]
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
