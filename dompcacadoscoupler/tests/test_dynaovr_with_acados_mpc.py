import copy

import numpy as np
from basiclibrary.system_utilities import create_logger
from do_mpc.data import Data
from dynamodel.dompc import do_mpc_helper as doh
from dynamodel.model_interface import Parameter
from dynaovr.receiver_modules.receiver_variables import ReceiverVariables
from dynaovrcontroller.flux_over_time import flux_over_time
from dynaovrcontroller.model_predictive_controller import (
    ControllerOptions, create_model_predictive_controller,
    extract_model_predicitve_controller)

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc

logger = create_logger(__name__)


def test_dynaovr_acados_mpc(duration_of_ramp: float = 10,
                            duration_of_flux_low: float = 10) -> None:
    variables = ReceiverVariables()
    # variables.p.air_return_ratio = 0.6
    # variables.p.solar_absorptivity = Parameter(0.6, hard_coded=True)
    variables.tvp.flux_solar = False
    variables.tvp.T_return_3 = False
    variables.tvp.T_ambient = False
    controller_options = ControllerOptions(scale_variables=False)

    # Shall MPC know about the tvp?
    controller_options.init_tvp_explicit = False

    # Set a realistic initial state
    variables.x.T_absorber_back = 1050
    variables.x.T_absorber_front = 1100

    # Set realistic flux (no impact if flux is treated as tvp)
    variables.p.flux_solar = lambda t: flux_over_time(
        t,
        duration_of_ramp=duration_of_ramp,
        duration_of_flux_low=duration_of_flux_low)

    if not controller_options.init_tvp_explicit:
        variables_for_mpc = copy.deepcopy(variables)
        variables_for_mpc.p.flux_solar = variables_for_mpc.p.flux_solar(0)
        # mpc = extract_model_predicitve_controller(variables_for_mpc)
        mpc, sx_variables = create_model_predictive_controller(variables)
        objective = sx_variables.x.T_absorber_front**2
        mpc.set_objective(objective, objective)
        mpc.setup()
        doh.init_variables(mpc, variables)
        mpc.set_initial_guess()
    else:
        # mpc = extract_model_predicitve_controller(variables)
        mpc, sx_variables = create_model_predictive_controller(variables)
        objective = sx_variables.x.T_absorber_front**2
        mpc.set_objective(objective, objective)
        mpc.setup()
        doh.init_variables(mpc, variables)
        mpc.set_initial_guess()
    set_acados_mpc(mpc)
    x_0 = np.vstack([mpc.x0[key] for key in mpc.x0.keys()])
    for _ in range(3):
        u_0 = mpc.make_step(x_0)


if __name__ == '__main__':
    test_dynaovr_acados_mpc()
