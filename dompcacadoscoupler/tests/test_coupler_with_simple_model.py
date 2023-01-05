from typing import Any, Dict, Tuple

import numpy as np
from do_mpc.controller import MPC
from do_mpc.model import Model

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc


def setup_simple_model() -> Model:
    model_type = 'continuous'  # either 'discrete' or 'continuous'
    model = Model(model_type)
    x = model.set_variable('_x', 'x')
    z = model.set_variable('_z', 'z')
    u = model.set_variable('_u', 'u')

    model.set_rhs('x', z + u)
    model.set_alg('z_alg', z - 1)
    model.setup()
    return model


def setup_simple_model_2() -> Model:
    model_type = 'continuous'
    model = Model(model_type)
    x = model.set_variable('_x', 'x')
    x_1 = model.set_variable('_x', 'x_1')
    u = model.set_variable('_u', 'u')
    model.set_rhs('x', x_1)
    model.set_rhs('x_1', (x - u)**2)
    model.setup()
    return model


def create_simple_mpc(model: Model) -> MPC:
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    mterm = model.x['x']**2
    lterm = model.x['x']**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    return mpc


def setup_simple_mpc(model: Model) -> MPC:
    mpc = create_simple_mpc(model)
    mpc.setup()
    return mpc


def compare_ipopt_and_acados(model: Model) -> Tuple:
    mpc = setup_simple_mpc(model)
    x0 = np.array([[0]])
    u_ipopt = mpc.make_step(x0)

    mpc = setup_simple_mpc(model)
    set_acados_mpc(mpc)
    x0 = np.array([[0]])
    u_acados = mpc.make_step(x0)
    return u_ipopt, u_acados


def test_compare_ipopt_and_acados() -> None:
    model = setup_simple_model()
    u_ipopt, u_acados = compare_ipopt_and_acados(model)
    np.testing.assert_allclose(u_acados, u_ipopt)


def test_compare_ipopt_and_acados_2() -> None:
    model = setup_simple_model_2()
    mpc = setup_simple_mpc(model)
    x0 = np.array([[1, 0]])
    u_ipopt = mpc.make_step(x0)

    mpc = setup_simple_mpc(model)
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL',
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1, 0]])
    u_acados = mpc.make_step(x0)
    np.testing.assert_allclose(u_acados, u_ipopt)


def test_mpc_with_explicit_runge_kutta() -> None:
    model = setup_simple_model_2()
    mpc = setup_simple_mpc(model)
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'ERK',
        'cost_type': 'EXTERNAL',
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1, 0]])
    u_acados = mpc.make_step(x0)
    np.testing.assert_allclose(u_acados, 1)


def test_mpc_with_t_step() -> None:
    t_step = 0.1
    model = setup_simple_model_2()

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
    np.testing.assert_allclose(u_acados, u_ipopt)


def setup_mpc_with_t_step(model: Model, t_step: float) -> MPC:
    mpc = create_simple_mpc(model)
    mpc.t_step = t_step
    mpc.setup()
    return mpc


if __name__ == '__main__':
    test_mpc_with_t_step()