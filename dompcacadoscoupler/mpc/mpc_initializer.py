from typing import Union

import casadi as cd
import numpy as np
from acados_template import AcadosOcpSolver
from do_mpc.controller import MPC


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
            if z_variables.size != 0:
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
    if z_init.size != 0:
        for stage in range(0, n_horizon):
            acados_solver.set(stage, 'z', z_init)
    for stage in range(0, n_horizon):
        acados_solver.set(stage, 'u', u_init)
