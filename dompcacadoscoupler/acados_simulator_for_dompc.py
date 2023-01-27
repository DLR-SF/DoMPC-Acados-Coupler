from typing import Any, Dict, Union

import casadi as cd
import matplotlib.pyplot as plt
import numpy as np
from acados_template import (AcadosModel, AcadosSim, AcadosSimOpts,
                             AcadosSimSolver)
from dynamodel.examples.pt1_model_coupling import (Simulator, create_pt2_model,
                                                   set_x_init)

from dompcacadoscoupler.model_converter import convert_to_acados_model


def set_acados_simulator(simulator: Simulator):
    acados_simulator = AcadosDompcSimulator(simulator)
    simulator.simulator = acados_simulator


class AcadosDompcSimulator:

    def __init__(self, simulator: Simulator) -> None:
        self.acados_integrator = convert_to_acados_simulator(simulator)

    def __call__(self, x0, z0, p) -> Dict[str, Union[np.ndarray, float]]:
        x0 = cd.vertcat(x0)
        z0 = cd.vertcat(z0)
        u0 = p['_u']
        p0 = p['_p']
        tvp0 = p['_tvp']
        p_total_0 = cd.vertcat(p0, tvp0)
        result_dict = simulate(self.acados_integrator, x0, z0, u0, p_total_0)

        return result_dict


def simulate(
    acados_integrator: AcadosSimSolver,
    x0: cd.DM,
    z0: cd.DM,
    u0: cd.DM,
    p0: cd.DM,
) -> Dict[str, Union[np.ndarray, float]]:
    acados_integrator.set('x', np.asarray(x0))
    if not z0.is_empty():
        acados_integrator.set('z', np.asarray(z0))
    acados_integrator.set('u', np.asarray(u0))
    acados_integrator.set('p', np.asarray(p0))
    # initialize IRK
    if acados_integrator.acados_sim.solver_options.integrator_type != 'ERK':
        nx = acados_integrator.acados_sim.dims.nx
        acados_integrator.set('xdot', np.zeros((nx,)))

    # solve
    status = acados_integrator.solve()
    if status != 0:
        raise RuntimeError(f'Acados returned status {status}.')
    # get solution
    result_x = acados_integrator.get('x')
    result_z = acados_integrator.get('z')
    result_dict = {}
    result_dict['xf'] = cd.DM(result_x)
    result_dict['zf'] = cd.DM(result_z)
    return result_dict


def convert_to_acados_simulator(simulator: Simulator) -> AcadosSimSolver:
    acados_model = convert_to_acados_model(simulator.model)
    acados_simulator = create_acados_simulator(acados_model)
    acados_simulator.solver_options = determine_simulator_options(simulator)
    acados_integrator = AcadosSimSolver(acados_simulator)
    return acados_integrator


def create_acados_simulator(acados_model: AcadosModel) -> AcadosSim:
    sim = AcadosSim()
    sim.model = acados_model
    # They are initialized with 0 to have any value. Otherwise, the solver
    # raises an error. The actual parameters are set later.
    parameter_shape = np.shape(acados_model.p)
    if len(parameter_shape) == 2 and parameter_shape[1] != 1:
        raise RuntimeError('Fatal error. Parameters are not in vector form.')
    sim.parameter_values = np.zeros(parameter_shape[0])
    return sim


def determine_simulator_options(simulator: Simulator) -> AcadosSimOpts:
    solver_options = AcadosSimOpts()
    if hasattr(simulator, 'acados_options'):
        dompc_options = simulator.acados_options  # type: ignore
    else:
        dompc_options = {}
    solver_options.T = simulator.t_step  # type: ignore
    for option_name, value in dompc_options.items():
        setattr(solver_options, option_name, value)
    if not simulator.model.z.cat.is_empty():
        # # From: https://github.com/acados/acados/issues/893
        solver_options.output_z = True
        solver_options.integrator_type = dompc_options.get(
            'integrator_type', 'IRK')
        if solver_options.integrator_type == 'ERK':
            raise ValueError(
                'Explicit Runge-Kutta methods can not be applied when there are algebraic variables.'
            )
    return solver_options