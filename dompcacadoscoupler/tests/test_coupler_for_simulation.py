import matplotlib.pyplot as plt
import numpy as np
from dynamodel.examples.pt1_model_coupling import (Simulator, create_pt2_model,
                                                   set_x_init)

from dompcacadoscoupler.acados_simulator_for_dompc import set_acados_simulator


def test_simulator_conversion() -> None:
    pt2_variables, pt2_model = create_pt2_model()
    pt2_model.setup()
    pt2_simulator = Simulator(pt2_model)
    pt2_simulator.set_param(t_step=1)
    set_x_init(pt2_simulator, pt2_variables)
    pt2_simulator.setup()
    pt2_simulator.set_p_fun(lambda t: pt2_simulator.get_p_template())
    set_acados_simulator(pt2_simulator)

    nx = pt2_model.x.shape[0]
    nu = pt2_model.u.shape[0]
    N = 10
    result_array = np.zeros((N + 1, nx))
    x0 = np.array([0, 0])
    u0 = np.array([[1]]).transpose()

    result_array[0, :] = x0

    for i in range(N):
        result = pt2_simulator.make_step(u0)

        result_array[i + 1, :] = np.asarray(result).squeeze()
    # Values taken from idas solver.
    expected_results = np.array([
        [0., 0.],
        [0.632, 0.264],
        [0.865, 0.594],
        [0.95, 0.801],
        [0.982, 0.908],
        [0.993, 0.96],
        [0.998, 0.983],
        [0.999, 0.993],
        [1., 0.997],
        [1., 0.999],
        [1., 1.],
    ])
    np.testing.assert_allclose(result_array, expected_results, atol=0.001)
    plot_results = False
    if plot_results:
        plt.plot(result_array)
        plt.show()


if __name__ == '__main__':
    test_simulator_conversion()