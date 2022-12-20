from typing import Any, Dict, Union

import casadi as cd
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
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
    # TODO: Set p.
    acados_integrator.set('x', np.asarray(x0))
    acados_integrator.set('z', np.asarray(z0))
    acados_integrator.set('u', np.asarray(u0))
    acados_integrator.set('p', np.asarray(p0))
    # initialize IRK
    sim = acados_integrator.acados_sim
    if sim.solver_options.integrator_type == 'IRK':
        nx = acados_integrator.acados_sim.dims.nx
        acados_integrator.set('xdot', np.zeros((nx,)))

        # solve
    status = acados_integrator.solve()
    if status != 0:
        raise Exception(f'acados returned status {status}.')
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
    acados_simulator.solver_options.T = simulator.t_step  # type: ignore
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
    # set options
    sim.solver_options.integrator_type = 'IRK'  #'IRK'
    # From: https://github.com/acados/acados/issues/893
    sim.solver_options.output_z = True
    # NOTE: I do not know what these options do.
    sim.solver_options.num_stages = 3
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 3  # for implicit integrator
    sim.solver_options.collocation_type = 'GAUSS_RADAU_IIA'

    return sim