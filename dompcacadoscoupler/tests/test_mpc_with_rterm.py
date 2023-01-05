import numpy as np
from do_mpc.controller import MPC
from do_mpc.model import Model

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc


def create_model() -> Model:
    model_type = 'continuous'
    model = Model(model_type)
    x_1 = model.set_variable('_x', 'x_1')
    dx_1 = model.set_variable('_x', 'dx_1')
    x_2 = model.set_variable('_x', 'x_2')

    u = model.set_variable('_u', 'u')

    model.set_rhs('x_1', dx_1)

    model.set_rhs('dx_1', x_1 - x_2)
    model.set_rhs('x_2', u - x_2)
    model.setup()
    return model


def create_pendulum_mpc() -> MPC:
    model = create_model()
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 1,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    x_1 = model.x['x_1']
    mterm = x_1**2
    lterm = x_1**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-2)

    mpc.setup()
    return mpc


def test_mpc_with_rterm() -> None:
    x0 = np.pi * np.array([1, 1, -1.5]).reshape(-1, 1)

    mpc = create_pendulum_mpc()
    u_ipopt = mpc.make_step(x0)
    x1_ipopt_solution, ipopt_cost = extract_solution_and_costs(mpc)

    mpc = create_pendulum_mpc()
    mpc.acados_options = {
        'qp_solver':
            'PARTIAL_CONDENSING_HPIPM',  #FULL_CONDENSING_QPOASES,PARTIAL_CONDENSING_HPIPM
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL'
    }
    set_acados_mpc(mpc)
    u_acados = mpc.make_step(x0)
    x1_acados_solution, acados_cost = extract_solution_and_costs(mpc)
    print(f'{x1_ipopt_solution=}')
    print(f'{x1_acados_solution=}')
    print(f'{ipopt_cost=}')
    print(f'{acados_cost=}')
    print(f'{u_ipopt=}')
    print(f'{u_acados=}')
    np.testing.assert_allclose(u_acados, u_ipopt, rtol=0.003)


def extract_solution_and_costs(mpc: MPC):
    x_solution = np.asarray(mpc.opt_x_num_unscaled['_x']).squeeze()
    u_solution = np.asarray(mpc.opt_x_num_unscaled['_u']).squeeze()
    x1_solution = x_solution[:, :, 0]
    values_at_sample_time = x1_solution[:, -1]
    penalty = mpc.rterm_factor['u']
    costs = np.mean(values_at_sample_time**2) + penalty * u_solution**2
    return x1_solution, costs


if __name__ == '__main__':
    test_mpc_with_rterm()