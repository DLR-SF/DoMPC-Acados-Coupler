import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def create_model() -> AcadosModel:
    model_name = 'minimal_example'
    # set up states & controls
    x = cd.SX.sym('x')
    u = cd.SX.sym('u')
    # xdot
    x_dot = cd.SX.sym('x_dot')

    f_expl = (x - u)**2
    f_impl = x_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.name = model_name

    return model


def test_minimal_mpc_example_with_erk() -> None:
    ocp = AcadosOcp()
    model = create_model()
    ocp.model = model

    Tf = 2
    N = 2

    # set dimensions
    ocp.dims.N = N

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.x**2
    ocp.model.cost_expr_ext_cost_e = model.x**2

    # set constraints
    ocp.constraints.x0 = np.array([0])

    # set options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    z_init = np.array([])
    n_horizon = acados_solver.acados_ocp.dims.N
    assert n_horizon
    # When this for loop is commented out, the code works.
    # BUG: This is a bug in acados.
    # for stage in range(0, n_horizon):
    #     acados_solver.set(stage, 'z', z_init)
    acados_solver.solve()


if __name__ == '__main__':
    test_minimal_mpc_example_with_erk()