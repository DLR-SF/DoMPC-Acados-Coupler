from typing import Any, Dict, Optional, Union
from warnings import warn

import casadi as cd
import colorama
import matplotlib.pyplot as plt
import numpy as np
from acados_template import (AcadosModel, AcadosOcp, AcadosOcpConstraints,
                             AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver,
                             AcadosSim, AcadosSimSolver)
from do_mpc.controller import MPC
from dynamodel.examples.pt1_model_coupling import create_pt2_model, set_x_init
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
        # x0 = cd.vertcat(x0)
        # z0 = cd.vertcat(z0)
        # u0 = p['_u']
        p0 = p['_p']
        tvp0 = p['_tvp']

        solve(self.acados_solver, p0, tvp0)
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


def solve(
    acados_solver: AcadosOcpSolver,
    p: cd.DM,
    tvp: cd.DM,
) -> None:
    set_p(acados_solver, p, tvp)

    # solve
    status = acados_solver.solve()
    if status != 0:
        raise Exception(f'acados returned status {status}.')


def set_p(
    acados_solver: AcadosOcpSolver,
    p: cd.DM,
    tvp: cd.DM,
) -> None:
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
        p_total = np.hstack((p, current_tvp))
        acados_solver.set(stage, 'p', p_total)


def init_variables(mpc: MPC, acados_solver: AcadosOcpSolver) -> None:
    x_init = np.array(mpc.x0.cat)
    z_init = np.array(mpc.z0.cat)
    u_init = np.array(mpc.u0.cat)
    for stage in range(0, acados_solver.acados_ocp.dims.N + 1):
        acados_solver.set(stage, 'x', x_init)
    for stage in range(0, acados_solver.acados_ocp.dims.N):
        acados_solver.set(stage, 'z', z_init)
    for stage in range(0, acados_solver.acados_ocp.dims.N):
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
    # The correct parameter values should be set later via the set function.
    parameter_shape = (np.array(mpc.model.p.shape) +
                       np.array(mpc.model.tvp.shape))
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
    cost = determine_external_costs(mpc, acados_model)
    # TODO: Introduce soft constraints.
    return cost


def determine_external_costs(mpc: MPC,
                             acados_model: AcadosModel) -> AcadosOcpCost:
    cost = AcadosOcpCost()
    cost.cost_type = 'EXTERNAL'
    cost.cost_type_e = 'EXTERNAL'
    acados_model.cost_expr_ext_cost = mpc.lterm
    acados_model.cost_expr_ext_cost_e = mpc.mterm
    return cost


def determine_quadratic_costs(mpc: MPC) -> AcadosOcpCost:
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
    raise NotImplementedError()

    cost = AcadosOcpCost()
    if not cd.hessian(mpc.lterm, mpc.model.p)[0].is_empty():
        raise NotImplementedError(
            'Parameter variables are currently not allowed in the cost function.'
        )
    # It is divided by two because due to differentiation of the quadratic costs a factor of 2 is added.
    cost.Vx = get_hessian_as_array(mpc.lterm, mpc.model.x) / 2
    cost.Vz = get_hessian_as_array(mpc.lterm, mpc.model.z) / 2
    cost.Vu = np.asarray(cd.diag(mpc.rterm_factor.cat))
    # Setting all variables to 0 to get the reference.
    y_ref = mpc.lterm_fun(0, 0, 0, 0, 0)
    cost.yref = np.asarray(y_ref)
    # Meyer term.
    cost.Vx_e = get_hessian_as_array(mpc.mterm, mpc.model.x) / 2
    terminal_y_ref = mpc.mterm_fun(0, 0, 0)
    cost.yref_e = np.asarray(terminal_y_ref)
    n_y = np.shape(cost.Vx)[0] + np.shape(cost.Vu)[0]
    #TODO: This is not correct yet.
    # NOTE: In the bicicyle example they use:
    # unscale = N / Tf
    # What is that for?
    cost.W = np.ones((n_y, n_y))
    cost.W_e = np.ones((n_y, n_y))
    return cost


def get_hessian_as_array(term: Union[cd.SX, cd.MX, cd.DM],
                         variables: Union[cd.SX, cd.MX, cd.DM]) -> np.ndarray:
    # The first output of the hessian function is the hessian the second the gradient.
    # Thus, index 0 is used.
    return np.asarray(cd.DM(cd.hessian(term, variables)[0]))


def determine_solver_options(mpc: MPC) -> AcadosOcpOptions:
    solver_options = AcadosOcpOptions()
    # set QP solver and integration
    solver_options.tf = mpc.t_step
    # solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    solver_options.nlp_solver_type = "SQP"
    # TODO: Only use exact when cost function of type external.
    solver_options.hessian_approx = 'EXACT'  # 'GAUSS_NEWTON', 'EXACT'
    solver_options.integrator_type = "IRK"
    solver_options.sim_method_num_stages = 4
    solver_options.sim_method_num_steps = 3

    # solver_options.qp_solver_tol_stat = 1e-2
    # solver_options.qp_solver_tol_eq = 1e-2
    # solver_options.qp_solver_tol_ineq = 1e-2
    # solver_options.qp_solver_tol_comp = 1e-2
    return solver_options


def determine_acados_constraints(mpc: MPC) -> AcadosOcpConstraints:
    constraints = AcadosOcpConstraints()
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
        warn(colored(f'Soft constraints are not supported yet.', 'yellow'),
             stacklevel=2)
    return constraints


def bound_not_set(bound: cd.DM) -> bool:
    is_not_set = cd.DM.is_empty(bound) or (bound == cd.DM.inf(1, 1) or
                                           bound == -1 * cd.DM.inf(1, 1))
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
    if value == cd.DM.inf(1, 1):
        value = cd.DM(inf)
    elif value == -1 * cd.DM.inf(1, 1):
        value = cd.DM(-inf)
    else:
        pass
    return value


def extract_result(acados_solver: AcadosOcpSolver,
                   n_total_collocation_points: int) -> Dict[str, Any]:
    n_horizon = acados_solver.acados_ocp.dims.N
    n_x = acados_solver.acados_ocp.dims.nx
    n_u = acados_solver.acados_ocp.dims.nu
    # The order must be kept otherwise the ravel function yields not the right order of the elements.
    result_x = np.empty((n_horizon + 1, n_total_collocation_points + 1, n_x))
    result_u = np.empty((n_horizon, n_u))
    for stage_index, stage in enumerate(range(n_horizon + 1)):
        stage_x = acados_solver.get(stage, 'x')
        for index_x, x in enumerate(stage_x):
            result_x[stage_index, :, index_x] = x
        # TODO: Add z.
        # result_z = acados_solver.get('z')
        result_z = np.array([0])
        if stage_index != n_horizon:
            stage_u = acados_solver.get(stage, 'u')
            result_u[stage_index, :] = stage_u
    result_optimization_variables = np.hstack(
        (result_x.ravel(), result_u.ravel()))
    result_dict = {}
    result_dict['x'] = cd.DM(result_optimization_variables)
    result_dict['z'] = cd.DM(result_z)
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