import numpy as np
from do_mpc.controller import MPC
from do_mpc.model import Model

from dompcacadoscoupler.acados_mpc_for_dompc import (determine_solver_options,
                                                     set_acados_mpc)


def run_mpc_conversion(with_acados: bool = True) -> None:
    from dynamodel.examples.pt1_model_coupling import create_pt2_model

    # TODO: Test with time varying parameter.
    pt2_variables, pt2_model = create_pt2_model()
    # This is just done to have more than one input.
    pt2_model.set_variable('_u', 'temp_u')
    # z = pt2_model.set_variable('_z', 'z')
    # pt2_model.set_alg('alg_0', z - pt2_model.x['pt1_1.x.state'])
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


def test_determine_solver_options():
    model_type = 'continuous'
    model = Model(model_type)
    x = model.set_variable('_x', 'x')
    u = model.set_variable('_u', 'u')
    model.set_rhs('x', u)
    model.setup()
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'GAUSS_NEWTON',
        'integrator_type': 'IRK',
        'sim_method_num_stages': 5,
        'sim_method_num_steps': 4
    }

    solver_options = determine_solver_options(mpc)
    assert solver_options.tf == 2
    assert solver_options.qp_solver == 'PARTIAL_CONDENSING_HPIPM'
    assert solver_options.nlp_solver_type == 'SQP'
    assert solver_options.hessian_approx == 'GAUSS_NEWTON'
    assert solver_options.integrator_type == 'IRK'
    assert solver_options.sim_method_num_stages == 5
    assert solver_options.sim_method_num_steps == 4

    # Test default values
    mpc.acados_options = {}
    solver_options = determine_solver_options(mpc)
    assert solver_options.tf == 2
    assert solver_options.qp_solver == 'FULL_CONDENSING_QPOASES'
    assert solver_options.nlp_solver_type == 'SQP'
    assert solver_options.hessian_approx == 'EXACT'
    assert solver_options.integrator_type == 'IRK'
    assert solver_options.sim_method_num_stages == 4
    assert solver_options.sim_method_num_steps == 1


if __name__ == '__main__':
    # run_mpc_conversion(with_acados=False)
    test_determine_solver_options()
    run_mpc_conversion(with_acados=True)
