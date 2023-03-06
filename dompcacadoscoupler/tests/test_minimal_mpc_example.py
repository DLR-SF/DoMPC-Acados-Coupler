import uuid
from enum import Enum, auto

import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def export_simple_model() -> AcadosModel:
    model_name = 'minimal_example' + str(uuid.uuid4()).replace('-', '_')
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


def export_simple_dae_model(scalar: float = 1) -> AcadosModel:
    model_name = 'minimal_example' + str(uuid.uuid4()).replace('-', '_')
    # set up states & controls
    x = cd.SX.sym('x') * scalar
    u = cd.SX.sym('u') * scalar
    p = cd.SX.sym('p') * scalar
    # xdot
    x_dot = cd.SX.sym('x_dot')
    # algebraic variables
    z = cd.SX.sym('z') * scalar
    # dynamics
    f_expl = (u - 1) / scalar
    f_impl = x_dot - f_expl
    alg = z - 4 + u
    f_impl = cd.vertcat(f_impl, alg)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x / scalar
    model.xdot = x_dot
    model.z = z / scalar
    model.u = u / scalar
    model.p = p / scalar
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
    ocp.cost.yref = np.array([1])
    ocp.cost.Vx_e = np.array([[1]])
    ocp.cost.yref_e = np.array([1])
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
    np.testing.assert_allclose(cost_value, 0, atol=1e-5)


def export_simple_array_dae_model(scalar: float = 1) -> AcadosModel:
    model_name = 'minimal_example' + str(uuid.uuid4()).replace('-', '_')
    # set up states & controls
    x = cd.SX.sym('x', (2, 1)) * scalar
    u = cd.SX.sym('u', (2, 1)) * scalar
    p = cd.SX.sym('p', (2, 1)) * scalar
    # xdot
    x_dot = cd.SX.sym('x_dot', (2, 1))
    # algebraic variables
    z = cd.SX.sym('z', (2, 1)) * scalar
    # dynamics
    f_expl = (u - p) / scalar
    f_impl = x_dot - f_expl
    alg = z - 4 + u
    f_impl = cd.vertcat(f_impl, alg)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x / scalar
    model.xdot = x_dot
    model.z = z / scalar
    model.u = u / scalar
    model.p = p / scalar
    # set model_name
    model.name = model_name

    return model


class CostType(Enum):
    NONLINEAR_LS = auto()
    NONLINEAR_LS_WITH_CORRECT_COST_FUNCTION = auto()
    EXTERNAL_COSTFUNCTION = auto()


def test_minimal_mpc_example_with_nonlinear_ls() -> None:
    # If I switch this value to False it does work.
    cost_type = CostType.EXTERNAL_COSTFUNCTION

    ocp = AcadosOcp()

    # set model
    model = export_simple_array_dae_model()
    ocp.model = model

    Tf = 1.0
    N = 2

    # set dimensions
    ocp.dims.N = N
    # The cost function should be (x - 1)**2
    # Because x starts at 1, the optimal solution should be u=1 which results
    # in a cost value of 0.
    # + 0.1 to prevent the derivate of the cost function with the squareroot
    # to converge to infinity.
    cost = cd.sum1((model.x - 1)**2) + cd.sum1(  # type: ignore
        (model.z - 3)**2) + 0.1  # type: ignore
    cost_e = cd.sum1((model.x - 1)**2) + 0.1  # type: ignore
    if cost_type == CostType.NONLINEAR_LS:
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.model.cost_y_expr = cd.sqrt(cost)
        ocp.model.cost_y_expr_e = cd.sqrt(cost_e)
        ocp.cost.W = np.eye(1)
        ocp.cost.W_e = np.eye(1)
        ocp.cost.yref = np.array([0])
        ocp.cost.yref_e = np.array([0])
    elif cost_type == CostType.EXTERNAL_COSTFUNCTION:
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.solver_options.hessian_approx = 'EXACT'
        # Squared, so that it is equal to the nonlinear ls formulation.
        ocp.model.cost_expr_ext_cost = cost
        ocp.model.cost_expr_ext_cost_e = cost_e
    elif cost_type == CostType.NONLINEAR_LS_WITH_CORRECT_COST_FUNCTION:
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        cost = cd.vertcat(model.x - 1, model.z - 3)
        cost_e = cd.vertcat(model.x - 1)
        ocp.model.cost_y_expr = cost
        ocp.model.cost_y_expr_e = cost_e
        ocp.cost.W = np.eye(np.shape(cost)[0])
        ocp.cost.W_e = np.eye(np.shape(cost_e)[0])
        ocp.cost.yref = np.zeros(np.shape(cost))
        ocp.cost.yref_e = np.zeros(np.shape(cost_e))
    # set constraints
    ocp.constraints.x0 = np.array([1.0, 1.0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = Tf
    # If I set it to [1, 1] it converges to the trivial solution.
    p_init = np.array([1, 2])
    ocp.parameter_values = p_init
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    ocp_solver.solve()
    cost_value = ocp_solver.get_cost()
    ocp_solver.print_statistics()
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
    print(f'{cost_value=}')
    assert cost_value < 1, f'{cost_value=} is not smaller than 1.'


def test_minimal_mpc_example_with_scaling() -> None:
    # create ocp object to formulate the OCP
    ocp = create_ocp_with_scaling()
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
    np.testing.assert_allclose(cost_value, 0, atol=1e-5)


def create_ocp_with_scaling() -> AcadosOcp:
    ocp = AcadosOcp()
    scalar = 10
    # set model
    model = export_simple_dae_model(scalar)
    ocp.model = model

    Tf = 1.0
    N = 2

    # set dimensions
    ocp.dims.N = N
    # TODO: Is the scalar in the cost function?
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = (model.x * scalar - 1)**2
    ocp.model.cost_expr_ext_cost_e = (model.x * scalar - 1)**2

    # set constraints
    lower_bound = 1 / scalar
    upper_bound = 5 / scalar
    index = 0
    ocp.constraints.x0 = np.array([1.0 / scalar])
    ocp.constraints.lbx = np.array([lower_bound])
    ocp.constraints.ubx = np.array([upper_bound])
    ocp.constraints.idxbx = np.array([index])
    ocp.constraints.lbx_e = np.array([lower_bound])
    ocp.constraints.ubx_e = np.array([upper_bound])
    ocp.constraints.idxbx_e = np.array([index])
    ocp.constraints.lbu = np.array([lower_bound])
    ocp.constraints.ubu = np.array([upper_bound])
    ocp.constraints.idxbu = np.array([index])
    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = Tf
    ocp.parameter_values = np.array([2])
    return ocp


if __name__ == '__main__':
    test_minimal_mpc_example_with_nonlinear_ls()