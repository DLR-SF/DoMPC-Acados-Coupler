import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def export_simple_model() -> AcadosModel:
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
    f_expl = 1 + z + p
    f_impl = x_dot - f_expl
    alg = z - 1
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


def test_with_algebraic_variable_in_cost_function() -> None:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_simple_model()
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
    ocp_solver.solve()


def test_minimal_mpc_example() -> None:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_simple_model()
    ocp.model = model

    Tf = 1.0
    N = 20

    # set dimensions
    ocp.dims.N = N

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.x**2 + model.u**2
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
    p_now = np.array([3])
    for stage in range(0, ocp_solver.acados_ocp.dims.N):
        ocp_solver.set(stage, 'p', p_now)
    ocp_solver.solve()


def export_simple_dae_model() -> AcadosModel:
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
    f_expl = u - 1
    f_impl = x_dot - f_expl
    alg = z - 4 + u
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


def test_minimal_mpc_example_with_linear_ls() -> None:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_simple_dae_model()
    ocp.model = model

    Tf = 1.0
    N = 2

    # set dimensions
    ocp.dims.N = N

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    # The cost function should be (x - 1)**2
    # Because x starts at 1, the optimal solution should be u=1 which results
    # in a cost value of 0.
    ocp.cost.Vx = np.array([[1]])
    ocp.cost.Vu = np.array([[0]])
    ocp.cost.Vz = np.array([[0]])
    ocp.cost.yref = np.array([[1]])
    ocp.cost.Vx_e = np.array([[1]])
    ocp.cost.yref_e = np.array([[1]])
    ocp.cost.W = np.eye(1)
    ocp.cost.W_e = np.eye(1)

    # set constraints
    ocp.constraints.x0 = np.array([1.0])

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
    ocp_solver.solve()
    cost_value = ocp_solver.get_cost()
    n_horizon: int = ocp_solver.acados_ocp.dims.N  # type: ignore
    result_x = []
    for stage_index in range(n_horizon + 1):
        stage_x = ocp_solver.get(stage_index, 'x')
        result_x.append(stage_x)
    print(result_x)
    result_u = []
    for stage_index in range(n_horizon):
        stage_u = ocp_solver.get(stage_index, 'u')
        result_u.append(stage_u)
    print(result_u)
    assert cost_value == 0, 'Something is wrong.'


if __name__ == '__main__':
    test_minimal_mpc_example_with_linear_ls()