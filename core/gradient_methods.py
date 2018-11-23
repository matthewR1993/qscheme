from core.sytem_setup import *
from core.squeezing import *


def gd_with_momentum(algo_params, funct_params):
    """
    A gradient descent method.
    """
    alpha = algo_params['alpha']
    target_prec = algo_params['target_prec']
    betta = algo_params['betta']
    start_point = algo_params['start_point']
    search_iter_max = algo_params['search_iter_max']

    free_par_keys = funct_params['free_par_keys']
    quantity = funct_params['target_quantity_min']
    det_event = funct_params['det_event']
    phase_diff = funct_params['phase']
    state = funct_params['input_state']

    bs_params_arr = np.zeros(search_iter_max + 1, dtype=dict)
    bs_params_arr[0] = start_point

    momentums = np.zeros(search_iter_max + 1, dtype=list)
    grads = np.zeros((search_iter_max + 1, len(free_par_keys)), dtype=np.complex)
    alphas = np.zeros(search_iter_max + 1, dtype=float)
    alphas[0] = alpha

    # Calculating gradients.
    for i in range(search_iter_max):
        print('step:', i)
        for j in range(len(free_par_keys)):
            bs_params_upp = bs_params_arr[i].copy()
            bs_params_upp[free_par_keys[j]] += alphas[i]
            bs_params_down = bs_params_arr[i].copy()
            bs_params_down[free_par_keys[j]] -= alphas[i]
            dm_up, _, _ = process_all(state, bs_params_upp, phase_diff=phase_diff, det_event=det_event)
            dm_down, _, _ = process_all(state, bs_params_down, phase_diff=phase_diff, det_event=det_event)
            if quantity is 'EPR_X':
                epr_x_up, _ = erp_squeezing_correlations(dm_up)
                epr_x_down, _ = erp_squeezing_correlations(dm_down)
                grads[i, j] = (epr_x_up - epr_x_down) / (2 * alphas[i])
                ret_val = epr_x_up
            elif quantity is 'EPR_P':
                _, epr_p_up = erp_squeezing_correlations(dm_up)
                _, epr_p_down = erp_squeezing_correlations(dm_down)
                grads[i, j] = (epr_p_up - epr_p_down) / (2 * alphas[i])
                ret_val = epr_p_up
            elif quantity is 'QUADR_X':
                dX_up, _ = squeezing_quadratures(dm_up, channel=1)
                dX_down, _ = squeezing_quadratures(dm_down, channel=1)
                grads[i, j] = (dX_up - dX_down) / (2 * alphas[i])
                ret_val = dX_up
            elif quantity is 'QUADR_P':
                _, dP_up = squeezing_quadratures(dm_up, channel=1)
                _, dP_down = squeezing_quadratures(dm_down, channel=1)
                grads[i, j] = (dP_up - dP_down) / (2 * alphas[i])
                ret_val = dP_up
            else:
                raise ValueError

        # print('grads:', grads[i, :])
        print('grads max:', np.real(np.max(grads[i, :])))
        print('func val:', np.real(ret_val))
        print('coord t1, t4:', bs_params_arr[i]['t1'], bs_params_arr[i]['t4'])
        # Build up the next step by adding gradient.
        bs_params_next = bs_params_arr[i].copy()
        for j in range(len(free_par_keys)):
            bs_params_next[free_par_keys[j]] -= alphas[i] * np.real(grads[i, j])
        bs_params_arr[i + 1] = bs_params_next

        if np.max(np.real(grads[i, :])) <= target_prec:
            return {
                'is_found': True,
                'step': i,
                'min_val': ret_val,
                'gradients': grads,
                'momentums': momentums,
                'alphas': alphas,
                'params_arr': bs_params_arr
            }

        alphas[i + 1] = alphas[i] / algo_params['alpha_scale']

    return {
        'is_found': False,
        'gradients': grads,
        'momentums': momentums,
        'alphas': alphas,
        'params_arr': bs_params_arr
        }
