import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def export_simple_inverse_model() -> AcadosModel:
    model_name = 'minimal_example'
    # set up states & controls
    x = cd.SX.sym('x')
    u = cd.SX.sym('u')
    p = cd.SX.sym('p')
    # xdot
    x_dot = cd.SX.sym('x_dot')
    # algebraic variables
    z = cd.SX.sym('z')
    # dynamics
    f_expl = 1 + 1 / z + 1 / p + 1 / u + 1 / x
    f_impl = x_dot - f_expl
    alg = 1 / z - 1
    f_impl = cd.vertcat(f_impl, alg)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.z = z
    model.u = u
    model.p = p
    # set model_name
    model.name = model_name

    return model


def test_acados_initialization() -> None:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_simple_inverse_model()
    ocp.model = model

    Tf = 1.0
    N = 20

    # set dimensions
    ocp.dims.N = N

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.x**2 + model.u**2 + model.z**2
    ocp.model.cost_expr_ext_cost_e = model.x**2

    # set constraints
    ocp.constraints.x0 = np.array([0.0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = Tf

    p_init = np.array([2])
    ocp.parameter_values = p_init
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    x_init = 1
    z_init = 1
    u_init = 1
    p_init = 1
    for stage in range(0, ocp_solver.acados_ocp.dims.N + 1):
        ocp_solver.set(stage, 'x', x_init)
    for stage in range(0, ocp_solver.acados_ocp.dims.N):
        ocp_solver.set(stage, 'z', z_init)
    for stage in range(0, ocp_solver.acados_ocp.dims.N):
        ocp_solver.set(stage, 'u', u_init)
    for stage in range(0, ocp_solver.acados_ocp.dims.N + 1):
        ocp_solver.set(stage, 'p', p_init)
    status = ocp_solver.solve()
    assert status == 0, f'acados returned status {status}.'


if __name__ == '__main__':
    test_acados_initialization()
