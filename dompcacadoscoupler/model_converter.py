from typing import Any

import casadi as cd
from acados_template import AcadosModel
from do_mpc.model import Model as DompcModel


def create_x_dot(x: Any) -> cd.SX:
    x_dot_array = []
    for name in x.keys():
        x_dot = cd.SX.sym(name + '_dot')
        x_dot_array.append(x_dot)
    return cd.vertcat(*x_dot_array)


def convert_to_acados_model(model: DompcModel,
                            name: str = 'temp') -> AcadosModel:
    if model.flags['setup'] != True:
        raise ValueError('Model must be setup beforehand.')
    acados_model = AcadosModel()
    acados_model.x = cd.vertcat(model.x)
    acados_model.u = cd.vertcat(model.u)
    acados_model.p = cd.vertcat(model.p, model.tvp)
    x_dot_rhs = cd.vertcat(model._rhs)
    x_dot = create_x_dot(model.x)
    acados_model.xdot = x_dot
    acados_model.f_impl_expr = x_dot - x_dot_rhs
    algebraic_equations = cd.vertcat(model._alg)
    acados_model.f_impl_expr = cd.vertcat(acados_model.f_impl_expr,
                                          algebraic_equations)
    acados_model.z = cd.vertcat(model.z)
    if algebraic_equations.is_empty() and acados_model.z.is_empty():
        # The explicit expression can only be provided for a ode not for a dae.
        # If you have a dae you can only use the implicit solver.
        acados_model.f_expl_expr = x_dot_rhs
    acados_model.name = name
    return acados_model