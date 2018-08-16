from core.sytem_setup import *


# Maximum number of iterations for a gradient descent..
SEARCH_ITER_MAX = 40


# TODO
def gradient_descent(quantity, start_point, state, phase_diff, det_event, delta=1e-7):
    raise NotImplementedError
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
    par_keys = bs_params.keys()

    # Start point.
    bs_params_arr = [bs_params]
    dm, _ = process_all(state, bs_params_arr[0], phase_diff=phase_diff, det_event=det_event)
    dm_array = [dm]
    t_array = [np.array([bs_params['t1'], bs_params['t2'], bs_params['t3'], bs_params['t4']])]

    # todo
    size = 4

    for i in range(1, SEARCH_ITER_MAX):
        # Calculating gradients.
        t_array[i] = np.array([t_array[i - 1][0], ])
        dm, _ = process_all(state, bs_params_arr[i], phase_diff=phase_diff, det_event=det_event)
        bs_params_upp = []

        for j in range(size):
            bs_params_upp = bs_params_arr[i].copy()
            bs_params_upp['t1'] = bs_params_upp['t1'] + 0
            dm_up1, _ = process_all(state, bs_params_arr[i], phase_diff=phase_diff, det_event=det_event)

        dm_array[i] = dm

        if quantity is 'epr x':
            pass

    return 0
