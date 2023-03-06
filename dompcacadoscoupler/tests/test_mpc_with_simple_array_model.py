from typing import Any, Dict, Literal, Tuple

import casadi as cd
import numpy as np
import pytest
from do_mpc.controller import MPC
from do_mpc.model import Model

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc


def setup_simple_model(symvar_type: Literal['SX', 'MX'] = 'SX') -> Model:
    model_type = 'continuous'
    model = Model(model_type, symvar_type)
    x = model.set_variable('_x', 'x', (2, 1))
    z = model.set_variable('_z', 'z', (2, 1))
    u = model.set_variable('_u', 'u', (2, 1))
    p = np.array([1, 2])
    model.set_rhs('x', u - p)
    model.set_alg('z_alg', z - 4 + u)
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
    mterm = cd.sum1(model.x['x']**2)
    lterm = cd.sum1(model.x['x']**2)
    mpc.set_objective(mterm=mterm, lterm=lterm)
    return mpc


def setup_simple_mpc(model: Model) -> MPC:
    mpc = create_simple_mpc(model)
    mpc.setup()
    return mpc


def compare_ipopt_and_acados(model: Model) -> Tuple:
    mpc = setup_simple_mpc(model)
    x0 = np.array([[0], [0]])
    u_ipopt = mpc.make_step(x0)

    mpc = setup_simple_mpc(model)
    set_acados_mpc(mpc)
    x0 = np.array([[0], [0]])
    u_acados = mpc.make_step(x0)
    return u_ipopt, u_acados


def test_compare_ipopt_and_acados() -> None:
    model = setup_simple_model()
    u_ipopt, u_acados = compare_ipopt_and_acados(model)
    np.testing.assert_allclose(u_acados, u_ipopt)


def test_compare_ipopt_and_acados_with_MX() -> None:
    model = setup_simple_model(symvar_type='MX')
    u_ipopt, u_acados = compare_ipopt_and_acados(model)
    np.testing.assert_allclose(u_acados, u_ipopt)


@pytest.mark.parametrize('symvar_type', ['SX', 'MX'])
def test_mpc_with_linear_cost_function(symvar_type) -> None:
    model = setup_simple_model(symvar_type)
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    lterm = cd.sum1((model.x['x'] - 1)**2) + cd.sum1((model.z['z'] - 3)**2)
    mterm = cd.sum1((model.x['x'] - 1)**2)
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.setup()
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'GAUSS_NEWTON',
        'integrator_type': 'IRK',
        'cost_type': 'LINEAR_LS',
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1], [1]])
    u_acados = mpc.make_step(x0)
    cost_value = mpc.S.acados_solver.get_cost()
    np.testing.assert_allclose(mpc.S.acados_solver.acados_ocp.cost.yref,
                               np.array([1, 1, 0, 0, 3, 3]))
    np.testing.assert_allclose(mpc.S.acados_solver.acados_ocp.cost.yref_e, 1)
    np.testing.assert_allclose(cost_value, 0.7, atol=1e-8)
    np.testing.assert_allclose(u_acados, [[1], [1.8]])


@pytest.mark.parametrize('symvar_type', ['SX', 'MX'])
def test_mpc_with_nonlinear_cost_function(symvar_type) -> None:
    model = setup_simple_model(symvar_type)
    mpc = MPC(model)
    # mpc.u0 = np.array([[1], [1.8]])
    setup_mpc = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    lterm = cd.sum1((model.x['x'] - 1)**2) + cd.sum1((model.z['z'] - 3)**2)
    mterm = cd.sum1((model.x['x'] - 1)**2)
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.setup()
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL',
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1], [1]])
    u_acados = mpc.make_step(x0)
    cost_value = mpc.S.acados_solver.get_cost()
    # np.testing.assert_allclose(cost_value, 0.7, atol=1e-8)
    # np.testing.assert_allclose(u_acados, [[1], [1.8]])


def create_mpc_simple_3(model: Model) -> MPC:
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    # lterm = (model.x['x'] - 1)**2 + (model.z['z'] - 3)**2
    # mterm = (model.x['x'] - 1)**2
    lterm = cd.sum1((model.x['x'] - 1)**2)
    mterm = cd.sum1((model.x['x'] - 1)**2)
    mpc.set_objective(lterm=lterm, mterm=mterm)
    return mpc


def create_mpc_simple_3_with_scaling(model: Model) -> MPC:
    mpc = create_mpc_simple_3(model)
    mpc.bounds['lower', '_x', 'x'] = 1
    mpc.bounds['upper', '_x', 'x'] = 5
    mpc.bounds['lower', '_u', 'u'] = 1
    mpc.bounds['upper', '_u', 'u'] = 5
    mpc.scaling['_x', 'x'] = 10
    mpc.scaling['_u', 'u'] = 10
    mpc.scaling['_z', 'z'] = 10
    mpc.x0 = np.array([[1], [1]])
    # mpc.u0 = 2
    # mpc.z0 = 1
    return mpc


def test_mpc_scaling() -> None:
    model = setup_simple_model()
    mpc = create_mpc_simple_3_with_scaling(model)
    mpc.setup()
    mpc.set_initial_guess()

    x0 = np.array([[1], [1]])
    u_ipopt = mpc.make_step(x0)

    mpc = create_mpc_simple_3_with_scaling(model)
    mpc.setup()
    mpc.set_initial_guess()
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL',
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1], [1]])
    u_acados = mpc.make_step(x0)
    np.testing.assert_allclose(u_acados, u_ipopt, rtol=1e-3)
    np.testing.assert_allclose(mpc.S.acados_solver.get_cost(), 0, atol=1e-6)


@pytest.mark.parametrize('symvar_type', ['SX', 'MX'])
def test_mpc_scaling_with_rterm(symvar_type) -> None:
    model = setup_simple_model(symvar_type)
    mpc = create_mpc_simple_3_with_scaling(model)

    mpc.set_rterm(u=100000)
    mpc.setup()
    mpc.u0 = np.array([[2], [2]])
    mpc.set_initial_guess()
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL',
    }
    set_acados_mpc(mpc)
    x0 = np.array([[1], [1]])
    u_acados = mpc.make_step(x0)
    cost_value = mpc.S.acados_solver.get_cost()
    print(cost_value)
    n_horizon: int = mpc.S.acados_solver.acados_ocp.dims.N  # type: ignore
    result_x = []
    for stage_index in range(n_horizon + 1):
        stage_x = mpc.S.acados_solver.get(stage_index, 'x')
        result_x.append(stage_x)
    print(result_x)
    np.testing.assert_allclose(cost_value, 5, atol=1e-3)
    np.testing.assert_allclose(u_acados, 2, atol=1e-3)


@pytest.mark.parametrize(('penalty', 'expected_cost_value'), [(10, 12.5),
                                                              (2, 6.125)])
def test_mpc_with_soft_constraint(penalty: int,
                                  expected_cost_value: float) -> None:
    x0 = np.array([[1], [1]])
    model = setup_simple_model()
    mpc = create_mpc_simple_3(model)

    mpc.set_nl_cons('soft_constraint',
                    model.u['u'],
                    0.5,
                    soft_constraint=True,
                    penalty_term_cons=penalty)
    mpc.setup()
    u_ipopt = mpc.make_step(x0)

    model = setup_simple_model()
    mpc = create_mpc_simple_3(model)

    mpc.set_nl_cons('soft_constraint',
                    model.u['u'],
                    0.5,
                    soft_constraint=True,
                    penalty_term_cons=penalty)
    mpc.setup()
    mpc.acados_options = {
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
        'nlp_solver_type': 'SQP',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL',
    }
    set_acados_mpc(mpc)
    u_acados = mpc.make_step(x0)
    cost_value = mpc.S.acados_solver.get_cost()
    print(cost_value)
    n_horizon: int = mpc.S.acados_solver.acados_ocp.dims.N  # type: ignore
    result_x = []
    for stage_index in range(n_horizon + 1):
        stage_x = mpc.S.acados_solver.get(stage_index, 'x')
        result_x.append(stage_x)
    print(result_x)
    result_s = []
    for stage_index in range(n_horizon):
        stage_s = mpc.S.acados_solver.get(stage_index, 'su')
        result_s.append(stage_s)
    print(result_s)
    np.testing.assert_allclose(cost_value, expected_cost_value, atol=1e-3)
    np.testing.assert_allclose(u_acados, u_ipopt, atol=1e-3)


if __name__ == '__main__':
    test_mpc_with_nonlinear_cost_function('SX')