from enum import Enum
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import casadi as cd
import colorama
import numpy as np
from acados_template import (AcadosModel, AcadosOcp, AcadosOcpConstraints,
                             AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver)
from do_mpc.controller import MPC
from termcolor import colored

from dompcacadoscoupler.model_converter import convert_to_acados_model

colorama.init()


def set_acados_mpc(mpc: MPC):
    """Overwrite the ipopt solver with the acados solver.

    Args:
        mpc (MPC): Do mpc object which was already setup.
    """
    acados_solver = AcadosDompcOcpSolver(mpc)
    # S is the ipopt solver in the dompc object.
    # This has to be overwritten with the acados solver.
    # It is on the lowest level. Therefore, it should not harm other dompc functionality.
    mpc.S = acados_solver


class AcadosDompcOcpSolver:
    """This class creates an acados mpc solver from a dompc object.
        It can be used to overwrite the internal dompc solver member self.S.
    """

    def __init__(self, mpc: MPC) -> None:
        self.acados_solver = convert_to_acados_mpc(mpc)
        init_variables(mpc, self.acados_solver)
        # Do not store the lagrange multiplier anymore.
        # They should not be important for dompc.
        # They could be extracted from acados. However, it was not obvious what corresponds to lam_g and lam_x.
        mpc.store_lagr_multiplier = False
        self.n_total_collocation_points = calculate_collocation_points(mpc)

    def __call__(self,
                 x0: Any,
                 lbx: Any,
                 ubx: Any,
                 lbg: Any,
                 ubg: Any,
                 p: Any,
                 lam_x0: Any = None,
                 lam_g0: Any = None) -> Dict[str, Union[np.ndarray, float]]:
        """This imitates the ipopt solver interface used in the dompc library.
            Apart from p the other variables are not used currently.
            lam_x0, lam_g0 are currently not used. For ipopt they were used to warm start the solver.
            For acados this warm starting should already happen internally.
            See: https://discourse.acados.org/t/initialization-of-mpc-steps-warm-start/171

        Args:
            x0 (Any): _description_
            lbx (Any): _description_
            ubx (Any): _description_
            lbg (Any): _description_
            ubg (Any): _description_
            p (Any): _description_

        Returns:
            Dict[str, Union[np.ndarray, float]]: A dictionary containing the keys x, g, lam_g and lam_x.
        """
        optimization_variable_guess = x0.cat
        solve(self.acados_solver, optimization_variable_guess, p,
              self.n_total_collocation_points)
        # get solution
        result_dict = extract_result(self.acados_solver,
                                     self.n_total_collocation_points)

        return result_dict

    def stats(self):
        return get_all_statistics(self.acados_solver)


def get_all_statistics(acados_solver: AcadosOcpSolver) -> Dict[str, Any]:
    solver_stats = {}
    for key in [
            'statistics', 'time_tot', 'time_lin', 'time_sim', 'time_sim_ad',
            'time_sim_la', 'time_qp', 'time_qp_solver_call', 'time_reg',
            'sqp_iter', 'residuals', 'qp_iter', 'alpha'
    ]:
        solver_stats[key] = acados_solver.get_stats(key)
    return solver_stats


class ReturnValues(Enum):
    ACADOS_SUCCESS = 0
    ACADOS_FAILURE = 1
    ACADOS_MAXITER = 2
    ACADOS_MINSTEP = 3
    ACADOS_QP_FAILURE = 4
    ACADOS_READY = 5


def solve(
    acados_solver: AcadosOcpSolver,
    optimization_variable_guess: cd.DM,
    p: Union[Dict[str, Any], Any],
    n_total_collocation_points: int,
) -> None:
    p0 = p['_p']
    u_previous = p['_u_prev']
    tvp0 = p['_tvp']
    x_at_step_0 = p['_x0']
    set_p(acados_solver, p0, tvp0, u_previous)
    set_x0(acados_solver, x_at_step_0)
    init_optimization_variables(acados_solver, optimization_variable_guess,
                                n_total_collocation_points)

    # solve
    status = acados_solver.solve()
    if status != 0:
        # NOTE: This link may give a hint for convergence problems.
        # TODO: What does alpha mean and why is it zero in the first iteration?
        raise Exception(f'acados returned status {ReturnValues(status).name}.')


def init_optimization_variables(acados_solver: AcadosOcpSolver, x_init: cd.DM,
                                n_total_collocation_points: int) -> None:
    x0 = np.asarray(x_init)
    n_horizon: int = acados_solver.acados_ocp.dims.N  # type: ignore
    n_x: int = acados_solver.acados_ocp.dims.nx  # type: ignore
    n_z: int = acados_solver.acados_ocp.dims.nz  # type: ignore
    n_u: int = acados_solver.acados_ocp.dims.nu  # type: ignore
    shape_x = (n_horizon + 1, n_total_collocation_points + 1, n_x)
    shape_z = (n_horizon, n_total_collocation_points, n_z)
    shape_u = (n_horizon, n_u)
    num_x = np.prod(shape_x)
    num_z = np.prod(shape_z)
    num_u = np.prod(shape_u)
    x_variables = np.reshape(x0[:num_x], shape_x)
    z_variables = np.reshape(x0[num_x:num_x + num_z], shape_z)
    u_variables = np.reshape(x0[num_x + num_z:num_x + num_z + num_u], shape_u)
    for stage_index in range(n_horizon + 1):
        acados_solver.set(stage_index, 'x', x_variables[stage_index, -1, :])
        if stage_index != n_horizon:
            acados_solver.set(stage_index, 'u', u_variables[stage_index, :])
            if z_variables:
                acados_solver.set(stage_index, 'z', z_variables[stage_index,
                                                                -1, :])


def set_x0(
    acados_solver: AcadosOcpSolver,
    x0: cd.DM,
) -> None:
    x0_casted = np.asarray(x0)
    acados_solver.set(0, "lbx", x0_casted)
    acados_solver.set(0, "ubx", x0_casted)


def set_p(
    acados_solver: AcadosOcpSolver,
    p: Union[cd.DM, np.ndarray],
    tvp: Union[cd.DM, np.ndarray],
    u_reference: Union[cd.DM, np.ndarray],
) -> None:
    """Set parameters of the optimization problem in the acados solver.

    Args:
        acados_solver: AcadosOcpSolver: The acados solver object.
        p: Union[cd.DM, np.ndarray]: The parameters of the optimization problem.
        tvp: Union[cd.DM, np.ndarray]: The time-varying parameters of the optimization problem.
        u_reference: Union[cd.DM, np.ndarray]: The reference control inputs for the optimization problem.
    """
    u_reference = np.asarray(u_reference).squeeze()
    p = np.asarray(p).squeeze()
    # Check if there are any time varying parameters.
    time_varying_parameters_exist = not tvp[0].is_empty()
    if p.size == 0 and not time_varying_parameters_exist:
        # No parameters in the model.
        return
    if time_varying_parameters_exist:
        tvp = np.asarray(tvp).squeeze()
    assert acados_solver.acados_ocp.dims.N
    for stage in range(0, acados_solver.acados_ocp.dims.N):
        # acados_solver.set(j, "yref", yref)
        if time_varying_parameters_exist:
            current_tvp = tvp[stage, :]
        else:
            current_tvp = np.array([])
        p_total = np.hstack((p, current_tvp, u_reference))
        acados_solver.set(stage, 'p', p_total)


def init_variables(mpc: MPC, acados_solver: AcadosOcpSolver) -> None:
    x_init = np.array(mpc.x0.cat)
    z_init = np.array(mpc.z0.cat)
    u_init = np.array(mpc.u0.cat)
    n_horizon = acados_solver.acados_ocp.dims.N
    assert n_horizon
    for stage in range(0, n_horizon + 1):
        acados_solver.set(stage, 'x', x_init)
    if z_init:
        for stage in range(0, n_horizon):
            acados_solver.set(stage, 'z', z_init)
    for stage in range(0, n_horizon):
        acados_solver.set(stage, 'u', u_init)


def convert_to_acados_mpc(mpc: MPC) -> AcadosOcpSolver:
    acados_model = convert_to_acados_model(mpc.model)
    acados_mpc = create_acados_mpc(mpc, acados_model)
    # Tf is the prediction horizon.
    acados_mpc.solver_options.tf = mpc.t_step * mpc.n_horizon  # type: ignore
    acados_solver = AcadosOcpSolver(acados_mpc)
    return acados_solver


def create_acados_mpc(mpc: MPC, acados_model: AcadosModel) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.constraints = determine_acados_constraints(mpc)
    ocp.solver_options = determine_solver_options(mpc)
    ocp.cost = determine_objective_function(mpc, acados_model)
    ocp.dims.N = mpc.n_horizon
    sanity_check_solver_options(ocp)
    # The correct parameter values should be set later via the set function.
    parameter_shape = np.shape(acados_model.p)
    ocp.parameter_values = np.ones(parameter_shape)
    return ocp


def determine_objective_function(mpc: MPC,
                                 acados_model: Optional[AcadosModel] = None
                                ) -> AcadosOcpCost:
    """Determine objective function from mpc object for acados.
        Be aware that the cost function in acados is defined quadratic not for each term 
        but for the addition of all terms. See: https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpCost
        Currently only quadratic costs are supported. Non quadratic costs may lead
        to unpredictable errors. 
    
    Args:
        mpc (MPC): Do mpc object.

    Raises:
        RuntimeError
        NotImplementedError

    Returns:
        AcadosOcpCost: Costs in acados format.
    """
    if not mpc.flags['set_objective']:
        raise RuntimeError('Objective function was not set in the mpc.')

    # cost = determine_quadratic_costs(mpc)
    assert acados_model
    if hasattr(mpc, 'acados_options'):
        cost_type = mpc.acados_options.get('cost_type', 'EXTERNAL')
    else:
        cost_type = 'EXTERNAL'

    if cost_type == 'EXTERNAL':
        cost = determine_external_costs(mpc, acados_model)
    elif cost_type == 'LINEAR_LS':
        cost = determine_linear_costs(mpc)
    else:
        raise ValueError(f'Cost type {cost_type} is not supported.')
    # TODO: Introduce soft constraints.
    return cost


def determine_external_costs(mpc: MPC,
                             acados_model: AcadosModel) -> AcadosOcpCost:
    cost = AcadosOcpCost()
    cost.cost_type = 'EXTERNAL'
    cost.cost_type_e = 'EXTERNAL'
    # For the rterm (rate of change of the inputs) there are two possibilities. Either you include udot as an additional state
    # and then include this in the cost function or you define a u_ref value in the cost function.
    # See: https://discourse.acados.org/t/implementing-rate-constraints-and-rate-costs/197/2
    # and see the acados paper for the second option.
    rterm = determine_rterm_by_reference(mpc, acados_model)
    acados_model.cost_expr_ext_cost = mpc.lterm + rterm  # type: ignore
    acados_model.cost_expr_ext_cost_e = mpc.mterm  # type: ignore
    return cost


def determine_rterm_by_reference(
        mpc: MPC, acados_model: AcadosModel) -> Union[cd.SX, cd.MX, cd.DM]:
    n_u = mpc.model.n_u
    attachment = '_ref'
    u_ref_list = []
    r_term = cd.DM(0)
    for u_name, penalty in dict(mpc.rterm_factor).items():
        if u_name == 'default':
            continue
        u = mpc.model.u[u_name]
        u_ref_name = u_name + attachment
        if isinstance(u, cd.MX):
            u_ref = cd.MX.sym(u_ref_name)
        elif isinstance(u, cd.SX):
            u_ref = cd.SX.sym(u_ref_name)
        else:
            raise ValueError(f'Type {type(u)} is not supported for the rterm.')
        r_term += penalty * (u - u_ref)**2
        u_ref_list.append(u_ref)

    acados_model.p = cd.vertcat(acados_model.p, *u_ref_list)
    return r_term


def determine_linear_costs(mpc: MPC) -> AcadosOcpCost:
    """Determines the cost with the linear standard formulization of the costs
        in acados. Be aware that the cost function in acados is defined 
        quadratic not for each term but for the addition of all terms. 
        See: https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpCost

    Args:
        mpc (MPC): Do mpc object with set objective function.

    Raises:
        NotImplementedError

    Returns:
        AcadosOcpCost: Acados cost term.
    """
    cost = AcadosOcpCost()
    cost.cost_type = 'LINEAR_LS'
    cost.cost_type_e = 'LINEAR_LS'
    p_term = cd.hessian(mpc.lterm, mpc.model.p)[0]
    if not (p_term.is_empty() or np.all(np.asarray(cd.DM(p_term)) == 0)):
        raise NotImplementedError(
            'Parameter variables are currently not allowed in the cost function.'
        )
    # It is divided by two because due to differentiation of the quadratic costs a factor of 2 is added.
    n_x = mpc.model.n_x
    n_z = mpc.model.n_z
    n_u = mpc.model.n_u
    n_w = n_x + n_z + n_u
    n_w_e = n_x
    Vx = get_hessian_as_array(mpc.lterm, mpc.model.x) / 2
    Vz = get_hessian_as_array(mpc.lterm, mpc.model.z) / 2
    Vu = np.asarray(cd.diag(mpc.rterm_factor.cat))
    cost.Vx = np.vstack((Vx, np.zeros((n_z + n_u, n_x))))
    cost.Vz = np.vstack((np.zeros((n_x, n_z)), Vz, np.zeros((n_u, n_z))))
    cost.Vu = np.vstack((np.zeros((n_x + n_z, n_u)), Vu))
    # Setting all variables to 0 to get the reference.
    # This only works if all terms are quadratic.
    lagrange_term_jacobian = mpc.lterm_fun.jacobian()
    jacobian_values = lagrange_term_jacobian(0, 0, 0, 0, 0, 0)
    y_ref = -jacobian_values / 2
    cost.yref = np.asarray(y_ref[:n_w]).T
    # Meyer term.
    cost.Vx_e = get_hessian_as_array(mpc.mterm, mpc.model.x) / 2
    meyer_term_jacobian = mpc.mterm_fun.jacobian()
    meyer_values = meyer_term_jacobian(0, 0, 0, 0)
    terminal_y_ref = -meyer_values / 2
    cost.yref_e = np.asarray(terminal_y_ref[:n_w_e]).T
    #TODO: This is not correct yet.
    # NOTE: In the bicicyle example they use:
    # unscale = N / Tf
    # What is that for?
    cost.W = np.ones((n_w, n_w))
    cost.W_e = np.ones((n_w_e, n_w_e))
    return cost


def get_hessian_as_array(term: Union[cd.SX, cd.MX, cd.DM],
                         variables: Union[cd.SX, cd.MX, cd.DM]) -> np.ndarray:
    # The first output of the hessian function is the hessian the second the gradient.
    # Thus, index 0 is used.
    hessian_index = 0
    return np.asarray(cd.DM(cd.hessian(term, variables)[hessian_index]))


def determine_solver_options(mpc: MPC) -> AcadosOcpOptions:
    solver_options = AcadosOcpOptions()
    # set QP solver and integration
    if hasattr(mpc, 'acados_options'):
        options = mpc.acados_options
    else:
        options = {}
    solver_options.tf = mpc.t_step
    solver_options.qp_solver = options.get('qp_solver',
                                           'FULL_CONDENSING_QPOASES')
    solver_options.nlp_solver_type = options.get('nlp_solver_type', "SQP")
    solver_options.hessian_approx = options.get('hessian_approx', 'EXACT')
    solver_options.integrator_type = options.get('integrator_type', "IRK")
    for option_name, value in options.items():
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


def determine_acados_constraints(mpc: MPC) -> AcadosOcpConstraints:
    constraints = AcadosOcpConstraints()
    x_init = np.array(mpc.x0.cat)
    constraints.x0 = x_init
    for variable_type in ['_x', '_z', '_u']:
        for index, variable_name in enumerate(mpc.model[variable_type].keys()):
            lower_bound = mpc.bounds['lower', variable_type, variable_name]
            upper_bound = mpc.bounds['upper', variable_type, variable_name]
            if bound_not_set(lower_bound) and bound_not_set(upper_bound):
                continue
            set_lower_bound(constraints, variable_type, lower_bound, index)
            set_upper_bound(constraints, variable_type, upper_bound, index)
    if len(mpc.slack_vars_list) > 1:
        # TODO: Implement soft constraints.
        warn(colored('Soft constraints are not supported yet.', 'yellow'),
             stacklevel=2)
    return constraints


def bound_not_set(bound: cd.DM) -> bool:
    casadi_inf = cd.DM.inf()
    is_not_set = cd.DM.is_empty(bound) or (bound == casadi_inf or
                                           bound == -1 * casadi_inf)
    return is_not_set


def set_lower_bound(constraints: AcadosOcpConstraints, variable_type: str,
                    lower_bound: Any, index: int):
    lower_bound = modify_when_infinity(lower_bound)
    if variable_type == '_x':
        constraints.lbx = np.append(constraints.lbx, lower_bound)
        constraints.idxbx = np.append(constraints.idxbx, index)
        constraints.lbx_e = np.append(constraints.lbx_e, lower_bound)
        constraints.idxbx_e = np.append(constraints.idxbx_e, index)
    elif variable_type == '_u':
        constraints.lbu = np.append(constraints.lbu, lower_bound)
        constraints.idxbu = np.append(constraints.idxbu, index)
    elif variable_type == '_z':
        # You may find some possibilities here: https://discourse.acados.org/t/python-interface-nonlinear-least-squares-and-constraints-on-algebraic-variables/44/7
        # Probably use lh for that.
        # NOTE: There is also a lsh bound for soft constraints.
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def set_upper_bound(constraints: AcadosOcpConstraints, variable_type: str,
                    upper_bound: Any, index: int):
    upper_bound = modify_when_infinity(upper_bound)
    if variable_type == '_x':
        constraints.ubx = np.append(constraints.ubx, upper_bound)
        constraints.ubx_e = np.append(constraints.ubx_e, upper_bound)
        if not np.isin(index, constraints.idxbx):
            constraints.idxbx = np.append(constraints.idxbx, index)
            constraints.idxbx_e = np.append(constraints.idxbx_e, index)
    elif variable_type == '_u':
        constraints.ubu = np.append(constraints.ubu, upper_bound)
        if not np.isin(index, constraints.idxbu):
            constraints.idxbu = np.append(constraints.idxbu, index)
    elif variable_type == '_z':
        # You may find some possibilities here: https://discourse.acados.org/t/python-interface-nonlinear-least-squares-and-constraints-on-algebraic-variables/44/7
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def modify_when_infinity(value: cd.DM) -> cd.DM:
    # Unfortunateley acados can apparently not handle infinity. Thus, the umber has to be converted to a really large number.
    # See: https://discourse.acados.org/t/one-sided-bounds-in-python-interface/69
    inf = 1e15
    casadi_inf = cd.DM.inf()
    if value == casadi_inf:
        value = cd.DM(inf)
    elif value == -1 * casadi_inf:
        value = cd.DM(-inf)
    else:
        pass
    return value


def extract_result(acados_solver: AcadosOcpSolver,
                   n_total_collocation_points: int) -> Dict[str, Any]:
    n_horizon: int = acados_solver.acados_ocp.dims.N  # type: ignore
    n_x: int = acados_solver.acados_ocp.dims.nx  # type: ignore
    n_z: int = acados_solver.acados_ocp.dims.nz  # type: ignore
    n_u: int = acados_solver.acados_ocp.dims.nu  # type: ignore
    # The order must be kept otherwise the ravel function yields not the right order of the elements.
    result_x = np.empty((n_horizon + 1, n_total_collocation_points + 1, n_x))
    # z has one collocation point less due to derivation of the collocation
    # optimization problem.
    # Also there are no z variables for the last stage.
    result_z = np.empty((n_horizon, n_total_collocation_points, n_z))
    result_u = np.empty((n_horizon, n_u))
    for stage_index in range(n_horizon + 1):
        stage_x = acados_solver.get(stage_index, 'x')
        for index_x, x in enumerate(stage_x):
            result_x[stage_index, :, index_x] = x
        if stage_index != n_horizon:
            stage_z = acados_solver.get(stage_index, 'z')
            for index_z, z in enumerate(stage_z):
                result_z[stage_index, :, index_z] = z
            stage_u = acados_solver.get(stage_index, 'u')
            result_u[stage_index, :] = stage_u
    unraveled_z_result = result_z.ravel()
    result_optimization_variables = np.hstack(
        (result_x.ravel(), unraveled_z_result, result_u.ravel()))
    result_dict = {}
    result_dict['x'] = cd.DM(result_optimization_variables)
    result_dict['z'] = cd.DM(unraveled_z_result)
    # They are probably not needed in the dompc object. They could be retrieved from acados, however this seems not to be obvious. Thus, they are set to an empty array.
    result_dict['g'] = cd.DM([])
    result_dict['lam_x'] = cd.DM([])
    result_dict['lam_g'] = cd.DM([])
    return result_dict


def calculate_collocation_points(mpc: MPC) -> int:
    deg = mpc.collocation_deg
    ni = mpc.collocation_ni
    n_total_collocation_points = (deg + 1) * ni
    return n_total_collocation_points