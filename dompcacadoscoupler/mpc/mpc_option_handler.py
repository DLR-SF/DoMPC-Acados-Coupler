from acados_template import AcadosOcp, AcadosOcpOptions
from do_mpc.controller import MPC


def determine_solver_options(mpc: MPC) -> AcadosOcpOptions:
    solver_options = AcadosOcpOptions()
    # set QP solver and integration
    if hasattr(mpc, 'acados_options'):
        options = mpc.acados_options
    else:
        options = {}
    # Tf is the prediction horizon.
    solver_options.tf = mpc.t_step * mpc.n_horizon  # type: ignore
    solver_options.qp_solver = options.get('qp_solver',
                                           'FULL_CONDENSING_QPOASES')
    solver_options.nlp_solver_type = options.get('nlp_solver_type', "SQP")
    solver_options.hessian_approx = options.get('hessian_approx', 'EXACT')
    solver_options.integrator_type = options.get('integrator_type', "IRK")
    for option_name, value in options.items():
        if option_name == 'cost_type':
            # The cost type was set in the cost object and is not set in the solver options
            continue
        setattr(solver_options, option_name, value)
    # In the context of numerical integration, a stage refers to a single evaluation of the derivative of the system being simulated at a particular time. A step is a single iteration of the integration algorithm and typically consists of multiple stages. For example, the often used Runge-Kutta 4 method has four stages per step.
    # solver_options.sim_method_num_stages = 4
    # solver_options.sim_method_num_steps = 1
    # solver_options.qp_solver_tol_stat = 1e-2
    # solver_options.qp_solver_tol_eq = 1e-2
    # solver_options.qp_solver_tol_ineq = 1e-2
    # solver_options.qp_solver_tol_comp = 1e-2
    return solver_options


def sanity_check_solver_options(ocp: AcadosOcp) -> None:
    external_cost_function = ocp.cost.cost_type == 'EXTERNAL' or ocp.cost.cost_type_0 == 'EXTERNAL'
    if external_cost_function and ocp.solver_options.hessian_approx != 'EXACT':
        raise ValueError(
            'If you want use the external cost function you must use the exact hessian approximation.'
        )
    if ocp.solver_options.integrator_type == 'ERK' and (
            not ocp.model.z.is_empty()):
        raise ValueError(
            'You can not use the explicit Runge-Kutta algorithm with algebraic variables.'
        )
