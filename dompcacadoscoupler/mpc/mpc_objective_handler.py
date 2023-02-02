from typing import Optional, Union

import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosOcpCost
from do_mpc.controller import MPC

from dompcacadoscoupler.mpc.mpc_scaler import scale_objective_function


def determine_objective_function(mpc: MPC,
                                 acados_model: Optional[AcadosModel] = None
                                ) -> AcadosOcpCost:
    """Determine objective function from mpc object for acados.
        Be aware that the cost function in acados is defined quadratic not for each term 
        but for the addition of all terms. See: https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpCost
    
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

    assert acados_model
    if hasattr(mpc, 'acados_options'):
        # Default is set to external.
        cost_type = mpc.acados_options.get('cost_type', 'EXTERNAL')
    else:
        cost_type = 'EXTERNAL'
    # This scales the mterm and the lterm.
    scale_objective_function(mpc)
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
    """
    Define the cost function for an optimal control problem (OCP) in the acados modeling and optimization library.

    The cost function is a combination of three terms: the lagrange term, the meyer term and the rate term. The rate term can be calculated using a reference value. The cost function is set as an attribute of the given `AcadosModel` object.

    Parameters:
        mpc (MPC): Object containing the lagrange term (`lterm`), meyer term (`mterm`) and rate term (`rterm`) for the OCP.
        acados_model (AcadosModel): Object representing the OCP model in acados.

    Returns:
        AcadosOcpCost: Object representing the cost function for the OCP.
    """
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
    """Determines the rate term (rterm) in the model predictive control (MPC) problem by reference.
    
    Args:
        mpc: The MPC object.
        acados_model: The Acados model object.
        
    Returns:
        The rate term in the MPC problem.
    """
    attachment = '_ref'
    u_ref_list = []
    r_term = cd.DM(0)
    for u_name, penalty in dict(mpc.rterm_factor).items():
        if u_name == 'default':
            continue
        u = mpc.model.u[u_name]
        u_ref_name = u_name + attachment
        if isinstance(u, cd.MX):
            u_ref = cd.MX.sym(u_ref_name, u.shape)
        elif isinstance(u, cd.SX):
            u_ref = cd.SX.sym(u_ref_name, u.shape)
        else:
            raise ValueError(f'Type {type(u)} is not supported for the rterm.')
        uncsaled_u = u * mpc.scaling['_u', u_name]
        delta_u = uncsaled_u - u_ref
        # Sum1 in case that u is an array.
        r_term += cd.sum1(penalty * (delta_u)**2)
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
    Vu = np.asarray(cd.diag(mpc.rterm_factor.cat))
    Vz = get_hessian_as_array(mpc.lterm, mpc.model.z) / 2
    cost.Vx = np.vstack((Vx, np.zeros((n_z + n_u, n_x))))
    cost.Vu = np.vstack((np.zeros((n_x, n_u)), Vu, np.zeros((n_z, n_u))))
    cost.Vz = np.vstack((np.zeros((n_x + n_u, n_z)), Vz))
    # Setting all variables to 0 to get the reference.
    # This only works if all terms are quadratic.
    lagrange_term_jacobian = mpc.lterm_fun.jacobian()
    jacobian_values = lagrange_term_jacobian(0, 0, 0, 0, 0, 0)
    y_ref = -jacobian_values / 2
    # The order of yref is [x,u,z].
    cost.yref = np.asarray(y_ref[:n_w]).ravel()
    # Meyer term.
    cost.Vx_e = get_hessian_as_array(mpc.mterm, mpc.model.x) / 2
    meyer_term_jacobian = mpc.mterm_fun.jacobian()
    meyer_values = meyer_term_jacobian(0, 0, 0, 0)
    terminal_y_ref = -meyer_values / 2
    cost.yref_e = np.asarray(terminal_y_ref[:n_w_e]).ravel()
    #TODO: This is not correct yet.
    # NOTE: In the bicicyle example they use:
    # unscale = N / Tf
    # What is that for?
    cost.W = np.eye(n_w)
    cost.W_e = np.eye(n_w_e)
    return cost


def get_hessian_as_array(term: Union[cd.SX, cd.MX, cd.DM],
                         variables: Union[cd.SX, cd.MX, cd.DM]) -> np.ndarray:
    # The first output of the hessian function is the hessian the second the gradient.
    # Thus, index 0 is used.
    hessian_index = 0
    return np.asarray(cd.DM(cd.hessian(term, variables)[hessian_index]))
