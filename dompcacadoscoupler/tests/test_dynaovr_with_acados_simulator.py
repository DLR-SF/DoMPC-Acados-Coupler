from time import perf_counter
from typing import Tuple

import numpy as np
from do_mpc.data import Data
from dynaovr.receiver_modules.receiver_variables import ReceiverVariables
from dynaovr.simulation.dompc_simulation import create_do_mpc_simulator

from dompcacadoscoupler.acados_simulator_for_dompc import set_acados_simulator


def test_dynaovr_with_acados_simulator() -> Tuple[ReceiverVariables, Data]:
    variables = ReceiverVariables()
    variables.p.comb_discretization = 0.999
    simulator = create_do_mpc_simulator(variables)
    set_acados_simulator(simulator)
    results = []
    results.append(np.array([[variables.x.T_absorber_front]]))
    u_0 = np.array([[0.0015]]).transpose()
    for i in range(9):
        simulated_output = simulator.make_step(u_0)
        T_absorber = simulated_output[0]
        results.append(T_absorber)
    expected_temperature = np.array([[293.15], [311.86933028], [330.40684548],
                                     [348.75951156], [366.92493047],
                                     [384.90106539], [402.68619029],
                                     [420.27886855], [437.67790983],
                                     [454.88229992]])
    results = np.vstack(results)
    np.testing.assert_allclose(results, expected_temperature, atol=2)
    return variables, simulator.data


def run_dynaovr_with_acados_simulator(
        use_acados_solver: bool = False) -> Tuple[ReceiverVariables, Data]:
    variables = ReceiverVariables()
    variables.p.comb_discretization = 0.999
    simulator = create_do_mpc_simulator(variables)
    if use_acados_solver:
        set_acados_simulator(simulator)
    u_0 = np.array([[0.0015]]).transpose()
    start = perf_counter()
    for i in range(10000):
        simulated_output = simulator.make_step(u_0)
    end = perf_counter()
    elapsed_time = end - start
    print(f'{elapsed_time=}')
    return variables, simulator.data


if __name__ == '__main__':
    test_dynaovr_with_acados_simulator()