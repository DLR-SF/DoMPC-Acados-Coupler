from typing import Any, Optional, Tuple, Union

import casadi as cd
import numpy as np
from acados_template import AcadosModel
from do_mpc.controller import MPC


def scale_objective_function(mpc: MPC) -> None:
    _x_unscaled, _u_unscaled, _z_unscaled, _p_unscaled, _tvp, _w = get_unscaled_variables(
        mpc)
    mpc.mterm = mpc.mterm_fun(_x_unscaled, _tvp, _p_unscaled)
    mpc.lterm = mpc.lterm_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp,
                              _p_unscaled)


def scale_acados_model(mpc: MPC, acados_model: AcadosModel):
    # Scale the odes
    # Scaled variables
    _rhs, _alg = scale_equations(mpc)
    # Adapt the acados model
    acados_model.f_expl_expr = _rhs
    f_impl_expr = acados_model.xdot - _rhs
    acados_model.f_impl_expr = cd.vertcat(f_impl_expr, _alg)


def scale_equations(mpc: MPC) -> Tuple[Any, Any]:
    # This code comes completely from dompc.
    _x_unscaled, _u_unscaled, _z_unscaled, _p_unscaled, _tvp, _w = get_unscaled_variables(
        mpc)

    # Create _rhs and _alg
    _rhs = mpc.model._rhs_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp,
                              _p_unscaled, _w)
    _alg = mpc.model._alg_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp,
                              _p_unscaled, _w)

    # Scale (only _rhs)
    # TODO: Shall I scale the algebraic equations?
    _rhs_scaled = _rhs / mpc._x_scaling.cat
    return _rhs_scaled, _alg


def get_unscaled_variables(mpc: MPC) -> Tuple:
    _x, _u, _z, _tvp, _p, _w = mpc.model['x', 'u', 'z', 'tvp', 'p', 'w']

    # Unscale variables
    _x_unscaled = _x * mpc._x_scaling.cat
    _u_unscaled = _u * mpc._u_scaling.cat
    _z_unscaled = _z * mpc._z_scaling.cat
    _p_unscaled = _p * mpc._p_scaling.cat
    return _x_unscaled, _u_unscaled, _z_unscaled, _p_unscaled, _tvp, _w
