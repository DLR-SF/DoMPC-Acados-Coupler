from enum import Enum
from typing import Any, Dict, Union

import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from do_mpc.controller import MPC

from dompcacadoscoupler.model_converter import convert_to_acados_model
from dompcacadoscoupler.mpc.mpc_constraint_handler import \
    determine_acados_constraints
from dompcacadoscoupler.mpc.mpc_initializer import (init_optimization_variables,
                                                    init_variables, set_p,
                                                    set_x0)
from dompcacadoscoupler.mpc.mpc_objective_handler import \
    determine_objective_function
from dompcacadoscoupler.mpc.mpc_option_handler import (
    determine_solver_options, sanity_check_solver_options)


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
            x0 (Any): Not used, just to maintain the interface.
            lbx (Any): Not used, just to maintain the interface.
            ubx (Any): Not used, just to maintain the interface.
            lbg (Any): Not used, just to maintain the interface.
            ubg (Any): Not used, just to maintain the interface.
            p (Any): Parameters for initialization.

        Returns:
            Dict[str, Union[np.ndarray, float]]: A dictionary containing the keys x, g, lam_g and lam_x.
        """
        optimization_variable_guess = x0.cat
        solve(self.acados_solver, optimization_variable_guess, p,
              self.n_total_collocation_points)
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
    # This is important for the first iteration
    # but may not be needed after the first iteration.
    init_optimization_variables(acados_solver, optimization_variable_guess,
                                n_total_collocation_points)

    status = acados_solver.solve()
    if status != 0:
        raise RuntimeError(
            f'acados returned status {ReturnValues(status).name}.')


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