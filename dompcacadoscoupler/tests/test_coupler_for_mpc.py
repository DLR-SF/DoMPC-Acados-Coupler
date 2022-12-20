import matplotlib.pyplot as plt
import numpy as np
from do_mpc.controller import MPC
from dynamodel.dompc import do_mpc_helper as doh
from dynamodel.examples.pt1_model_coupling import (Simulator, create_pt2_model,
                                                   set_x_init)

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc
from dompcacadoscoupler.acados_simulator_for_dompc import set_acados_simulator


def run_mpc_conversion(with_acados: bool = True) -> None:
    # TODO: Test with time varying parameter.
    pt2_variables, pt2_model = create_pt2_model()
    pt2_model.setup()
    mpc = MPC(pt2_model)
    setup_mpc = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    mterm = pt2_model.x['pt1_1.x.state']**2
    lterm = pt2_model.x['pt1_1.x.state']**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.bounds['lower', '_x', 'pt1_1.x.state'] = 0
    mpc.bounds['upper', '_x', 'pt1_1.x.state'] = 10
    mpc.bounds['lower', '_u', 'pt1_1.u.setpoint'] = 0
    mpc.bounds['upper', '_u', 'pt1_1.u.setpoint'] = 10
    # mpc.set_nl_cons('soft_constraint_0',
    #                 pt2_model.x['pt1_2.x.state'],
    #                 2,
    #                 soft_constraint=True)
    # set_x_init(mpc, pt2_variables)
    mpc.setup()
    # mpc.set_p_fun(lambda t: mpc.get_p_template())
    if with_acados:
        set_acados_mpc(mpc)
    x0 = np.array([0, 0])
    mpc.x0 = x0
    mpc.set_initial_guess()
    result_array = []
    result_array.append(x0)
    N = 10
    for i in range(N):
        result = mpc.make_step(x0)
        result_array.append(result)

    # Values taken from idas solver.
    # expected_results = np.array([
    #     [0., 0.],
    #     [0.632, 0.264],
    #     [0.865, 0.594],
    #     [0.95, 0.801],
    #     [0.982, 0.908],
    #     [0.993, 0.96],
    #     [0.998, 0.983],
    #     [0.999, 0.993],
    #     [1., 0.997],
    #     [1., 0.999],
    #     [1., 1.],
    # ])
    # np.testing.assert_allclose(result_array, expected_results, atol=0.001)
    # plot_results = True
    # if plot_results:
    #     plt.plot(result_array)
    #     plt.show()


if __name__ == '__main__':
    # run_mpc_conversion(with_acados=False)
    run_mpc_conversion(with_acados=True)
