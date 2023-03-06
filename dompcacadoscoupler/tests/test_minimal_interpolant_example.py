import uuid
from time import perf_counter

import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from do_mpc.controller import MPC
from do_mpc.model import Model

rng = np.random.default_rng(seed=42)


def create_interpolant() -> cd.Function:
    xgrid = np.linspace(-5, 5, 12)
    ygrid = np.linspace(-4, 4, 10)
    X, Y = np.meshgrid(xgrid, ygrid, indexing='ij')
    R = np.sqrt(5 * X**2 + Y**2) + rng.random(1)
    data = np.sin(R) / R
    data_flat = data.ravel(order='F')
    name = 'interpolant_' + str(uuid.uuid4()).replace('-', '_')
    lut = cd.interpolant(name, 'bspline', [xgrid, ygrid], data_flat)
    return lut


def export_simple_acados_model(n_interpolants: int = 2,
                               n_repititions: int = 1) -> AcadosModel:
    model_name = 'minimal_example' + str(uuid.uuid4()).replace('-', '_')
    # set up states & controls
    x = cd.MX.sym('x')
    u = cd.MX.sym('u')
    # xdot
    x_dot = cd.MX.sym('x_dot')
    # dynamics
    f_expl = 0
    for _ in range(n_interpolants):
        interpolant = create_interpolant()
        for index in range(n_repititions):
            f_expl = f_expl + interpolant(cd.vertcat(u, index))
    f_impl = x_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u
    # set model_name
    model.name = model_name

    return model


def export_simple_dompc_model(n_interpolants: int = 2,
                              n_repititions: int = 1) -> AcadosModel:
    # set up states & controls
    model = Model('continuous', 'MX')
    u = model.set_variable('_u', 'u')
    x = model.set_variable('_x', 'x')
    # dynamics
    f_expl = 0
    for _ in range(n_interpolants):
        interpolant = create_interpolant()
        for index in range(n_repititions):
            f_expl = f_expl + interpolant(cd.vertcat(u, index / n_repititions))
    model.set_rhs('x', f_expl)
    model.setup()
    return model


def test_with_interpolants(n_interpolants: int = 2,
                           n_repititions: int = 1) -> None:
    # create ocp object to formulate the OCP
    dompc_time = create_dompc_mpc(n_interpolants, n_repititions)
    acados_time = create_acados_mpc(n_interpolants, n_repititions)
    print(f'{dompc_time=}')
    print(f'{acados_time=}')


def create_dompc_mpc(n_interpolants: int = 2, n_repititions: int = 1) -> float:
    model = export_simple_dompc_model(n_interpolants, n_repititions)
    start = perf_counter()
    mpc = MPC(model)
    lterm = model.x['x']**2 + model.u['u']**2
    mterm = model.x['x']**2
    mpc.set_objective(mterm, lterm)
    dompc_options = {
        'n_horizon': 2,
        't_step': 1,
        'n_robust': 0,
    }
    mpc.set_param(**dompc_options)
    mpc.setup()
    end = perf_counter()
    elapsed_time = end - start
    return elapsed_time


def create_acados_mpc(n_interpolants: int = 2, n_repititions: int = 1) -> float:
    ocp = AcadosOcp()

    model = export_simple_acados_model(n_interpolants, n_repititions)
    ocp.model = model

    ocp.dims.N = 20
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.x**2 + model.u**2
    ocp.model.cost_expr_ext_cost_e = model.x**2
    ocp.solver_options.tf = 1.0

    start = perf_counter()
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    end = perf_counter()
    elapsed_time = end - start
    return elapsed_time


if __name__ == '__main__':
    test_with_interpolants(n_interpolants=200, n_repititions=5)
