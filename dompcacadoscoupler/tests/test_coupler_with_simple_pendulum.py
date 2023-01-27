import casadi as cd
import numpy as np
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc


def create_pendulum_model() -> Model:
    model_type = 'continuous'
    model = Model(model_type)
    x_1 = model.set_variable('_x', 'x_1')
    dx_1 = model.set_variable('_x', 'dx_1')
    u = model.set_variable('_u', 'u')

    model.set_rhs('x_1', dx_1)
    model.set_rhs('dx_1', x_1 - u)
    model.setup()
    return model


def create_pendulum_mpc(model: Model) -> MPC:
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 1,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
        'collocation_deg': 4
    }
    mpc.set_param(**setup_mpc)
    x_1 = model.x['x_1']
    mterm = x_1**2
    lterm = x_1**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    # mpc.set_rterm(u=1e-2)

    return mpc


def test_pendulum_mpc_without_array() -> None:
    x0 = np.array([1, 0]).reshape(-1, 1)
    model = create_pendulum_model()

    mpc = create_pendulum_mpc(model)
    mpc.setup()
    u_ipopt = mpc.make_step(x0)

    mpc = create_pendulum_mpc(model)
    mpc.setup()
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
    simulator = Simulator(model)
    simulator.x0 = x0
    simulator.set_param(t_step=1)
    simulator.setup()
    new_x = simulator.make_step(u_acados)
    np.testing.assert_allclose(new_x[0], 0, atol=1e-7)
    np.testing.assert_allclose(u_ipopt, u_acados, rtol=1e-5)


def test_mpc_with_t_step() -> None:
    t_step = 0.1
    model = create_pendulum_model()

    mpc = setup_mpc_with_t_step(model, t_step)
    x0 = np.array([[1, 0]])
    u_ipopt = mpc.make_step(x0)

    mpc = setup_mpc_with_t_step(model, t_step)
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL',
        'sim_method_num_steps': 3,
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1, 0]])
    u_acados = mpc.make_step(x0)
    np.testing.assert_allclose(u_acados, u_ipopt, rtol=0.0003)


def setup_mpc_with_t_step(model: Model, t_step: float) -> MPC:
    mpc = create_pendulum_mpc(model)
    mpc.t_step = t_step
    mpc.setup()
    return mpc


if __name__ == '__main__':
    test_pendulum_mpc_without_array()