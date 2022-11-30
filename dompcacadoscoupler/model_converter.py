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
    acados_model.p = cd.vertcat(model.p)
    acados_model.f_expl_expr = cd.vertcat(model._rhs)
    x_dot = create_x_dot(model.x)
    acados_model.xdot = x_dot
    acados_model.f_impl_expr = x_dot - acados_model.f_expl_expr
    acados_model.z = cd.vertcat(model.z)
    acados_model.name = name
    return acados_model