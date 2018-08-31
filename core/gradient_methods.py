from core.sytem_setup import *
from core.squeezing import *


def gradient_descent(algo_params, funct_params):
    '''
    A gradient descent method with the momentum.
    '''
    delta_t = algo_params['alpha']
    prec_t = algo_params['target_prec']
    betta = algo_params['betta']
    start_point = algo_params['start_point']
    par_keys = algo_params['par_keys']  # ['t1', 't2', 't3', 't4']
    fixed_params = algo_params['fixed_parameters']  # [{'t2': 0.5}, {'t3': 0.5}]
    search_iter_max = algo_params['search_iter_max']  # 40

    quantity = funct_params['min_quantity']
    det_event = funct_params['det_event']
    phase_diff = funct_params['phase']
    state = funct_params['input_state']

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
