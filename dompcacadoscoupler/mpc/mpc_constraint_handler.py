from typing import Any
from warnings import warn

import casadi as cd
import colorama
import numpy as np
from acados_template import AcadosOcpConstraints
from do_mpc.controller import MPC
from termcolor import colored

colorama.init()


def determine_acados_constraints(mpc: MPC) -> AcadosOcpConstraints:
    constraints = AcadosOcpConstraints()
    x_init = np.array(mpc.x0.cat / mpc._x_scaling)
    constraints.x0 = x_init.ravel()
    for variable_type in ['_x', '_z', '_u']:
        valid_labels = mpc.model[variable_type].labels()
        if 'default' in valid_labels:
            valid_labels.remove('default')
        for index, variable_label in enumerate(valid_labels):
            variable_name, variable_index = variable_label[1:-1].split(',')
            variable_index = int(variable_index)
            lower_bound = mpc.bounds['lower', variable_type, variable_name,
                                     variable_index]
            upper_bound = mpc.bounds['upper', variable_type, variable_name,
                                     variable_index]
            # Scale the bounds accordingly.
            scalar = mpc.scaling[variable_type, variable_name, variable_index]
            scaled_lower_bound = lower_bound / scalar
            scaled_upper_bound = upper_bound / scalar
            if bound_not_set(lower_bound) and bound_not_set(upper_bound):
                continue
            set_lower_bound(constraints, variable_type, scaled_lower_bound,
                            index)
            set_upper_bound(constraints, variable_type, scaled_upper_bound,
                            index)
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
    # Unfortunateley acados can apparently not handle infinity. Thus, the number has to be converted to a really large number.
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
