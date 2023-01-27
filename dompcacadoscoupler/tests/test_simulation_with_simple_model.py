from typing import Any, Dict, Tuple

import numpy as np
from do_mpc.model import Model
from do_mpc.simulator import Simulator

from dompcacadoscoupler.acados_simulator_for_dompc import set_acados_simulator


def setup_simple_model() -> Model:
    model_type = 'continuous'
    model = Model(model_type)
    x = model.set_variable('_x', 'x')
    z = model.set_variable('_z', 'z')
    z_2 = model.set_variable('_z', 'z_2')
    u = model.set_variable('_u', 'u')
    p = model.set_variable('_p', 'p')

    model.set_rhs('x', z + u + p)
    model.set_alg('z_alg_0', z - 1)
    model.set_alg('z_alg_1', z_2 - 1)
    model.setup()
    return model


def create_simple_simulator(model: Model) -> Simulator:
    simulator = Simulator(model)
    simulator.set_param(t_step=1, integration_tool='idas')
    p_template = simulator.get_p_template()

    def p_fun(t_now):
        p_template['p'] = 1
        return p_template

    simulator.set_p_fun(p_fun)
    simulator.x0 = 1
    return simulator


def setup_simple_simulator() -> Simulator:
    model = setup_simple_model()
    simulator = create_simple_simulator(model)
    simulator.setup()
    return simulator


def test_simulator() -> None:
    simulator = setup_simple_simulator()
    simulator.acados_options = {'integrator_type': 'IRK'}  # type: ignore
    set_acados_simulator(simulator)
    x_result = []
    u0 = np.array([[0]])
    for _ in range(10):
        next_x = simulator.make_step(u0)
        x_result.append(next_x)
    expected_x = np.array([3., 5., 7., 9., 11., 13., 15., 17., 19., 21.])
    np.testing.assert_allclose(np.asarray(x_result).squeeze(), expected_x)


if __name__ == '__main__':
    test_simulator()