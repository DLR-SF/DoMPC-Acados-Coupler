import casadi as cd
import numpy as np
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator

from dompcacadoscoupler.acados_mpc_for_dompc import set_acados_mpc


def create_pendulum_model(with_array: bool = True) -> Model:
    model_type = 'continuous'
    model = Model(model_type)
    phi_1 = model.set_variable('_x', 'phi_1', shape=(1, 1))
    phi_2 = model.set_variable('_x', 'phi_2', shape=(1, 1))
    phi_3 = model.set_variable('_x', 'phi_3', shape=(1, 1))
    if with_array:
        dphi = model.set_variable('_x', 'dphi', shape=(3, 1))
    else:
        dphi_1 = model.set_variable('_x', 'dphi_1')
        dphi_2 = model.set_variable('_x', 'dphi_2')
        dphi_3 = model.set_variable('_x', 'dphi_3')
        dphi = [dphi_1, dphi_2, dphi_3]
    phi_m_1_set = model.set_variable('_u', 'phi_m_1_set')
    phi_m_2_set = model.set_variable('_u', 'phi_m_2_set')
    phi_1_m = model.set_variable('_x', 'phi_1_m', shape=(1, 1))
    phi_2_m = model.set_variable('_x', 'phi_2_m', shape=(1, 1))
    Theta_1 = model.set_variable('parameter', 'Theta_1')
    Theta_2 = model.set_variable('parameter', 'Theta_2')
    Theta_3 = model.set_variable('parameter', 'Theta_3')
    c = np.array([2.697, 2.66, 3.05, 2.86]) * 1e-3
    d = np.array([6.78, 8.01, 8.82]) * 1e-5
    model.set_rhs('phi_1', dphi[0])
    model.set_rhs('phi_2', dphi[1])
    model.set_rhs('phi_3', dphi[2])

    dphi_next = cd.vertcat(
        -c[0] / Theta_1 * (phi_1 - phi_1_m) - c[1] / Theta_1 * (phi_1 - phi_2) -
        d[0] / Theta_1 * dphi[0],
        -c[1] / Theta_2 * (phi_2 - phi_1) - c[2] / Theta_2 * (phi_2 - phi_3) -
        d[1] / Theta_2 * dphi[1],
        -c[2] / Theta_3 * (phi_3 - phi_2) - c[3] / Theta_3 * (phi_3 - phi_2_m) -
        d[2] / Theta_3 * dphi[2],
    )
    if with_array:
        model.set_rhs('dphi', dphi_next)
    else:
        model.set_rhs('dphi_1', dphi_next[0])
        model.set_rhs('dphi_2', dphi_next[1])
        model.set_rhs('dphi_3', dphi_next[2])
    tau = 1e-2
    model.set_rhs('phi_1_m', 1 / tau * (phi_m_1_set - phi_1_m))
    model.set_rhs('phi_2_m', 1 / tau * (phi_m_2_set - phi_2_m))
    model.setup()
    return model


def create_pendulum_mpc(with_array: bool = True) -> MPC:
    model = create_pendulum_model(with_array)
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': 20,
        't_step': 0.1,
        'n_robust': 0,
        'store_full_solution': True,
        'collocation_deg': 4
    }
    mpc.set_param(**setup_mpc)
    phi_1 = model.x['phi_1']
    phi_2 = model.x['phi_2']
    phi_3 = model.x['phi_3']
    mterm = phi_1**2 + phi_2**2 + phi_3**2
    lterm = phi_1**2 + phi_2**2 + phi_3**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(phi_m_1_set=1e-2, phi_m_2_set=1e-2)
    mpc.bounds['lower', '_x', 'phi_1'] = -2 * np.pi
    mpc.bounds['lower', '_x', 'phi_2'] = -2 * np.pi
    mpc.bounds['lower', '_x', 'phi_3'] = -2 * np.pi
    mpc.bounds['upper', '_x', 'phi_1'] = 2 * np.pi
    mpc.bounds['upper', '_x', 'phi_2'] = 2 * np.pi
    mpc.bounds['upper', '_x', 'phi_3'] = 2 * np.pi
    mpc.bounds['lower', '_u', 'phi_m_1_set'] = -2 * np.pi
    mpc.bounds['lower', '_u', 'phi_m_2_set'] = -2 * np.pi
    mpc.bounds['upper', '_u', 'phi_m_1_set'] = 2 * np.pi
    mpc.bounds['upper', '_u', 'phi_m_2_set'] = 2 * np.pi
    mpc.scaling['_x', 'phi_1'] = 2
    mpc.scaling['_x', 'phi_2'] = 2
    mpc.scaling['_x', 'phi_3'] = 2
    p_template = mpc.get_p_template(n_combinations=1)

    def p_fun(t_now):
        p_template['_p', 0] = [2.25e-4, 2.25e-4, 2.25e-4]
        return p_template

    mpc.set_p_fun(p_fun)
    mpc.setup()
    x0 = np.pi * np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1, 1)
    mpc.x0 = x0
    mpc.set_initial_guess()
    return mpc


def create_pendulum_simulator(with_array: bool = True) -> Simulator:
    model = create_pendulum_model(with_array)
    simulator = Simulator(model)
    simulator.set_param(t_step=0.1)
    p_template = simulator.get_p_template()

    def p_fun(t_now):
        p_template['Theta_1'] = 2.25e-4
        p_template['Theta_2'] = 2.25e-4
        p_template['Theta_3'] = 2.25e-4
        return p_template

    simulator.set_p_fun(p_fun)
    simulator.setup()
    x0 = np.pi * np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1, 1)
    simulator.x0 = x0
    return simulator


def run_pendulum_mpc_without_array() -> None:
    x0 = np.pi * np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1, 1)

    mpc = create_pendulum_mpc(with_array=False)
    u_ipopt = mpc.make_step(x0)

    mpc = create_pendulum_mpc(with_array=False)
    mpc.acados_options = {
        'qp_solver':
            'PARTIAL_CONDENSING_HPIPM',  #FULL_CONDENSING_QPOASES,PARTIAL_CONDENSING_HPIPM
        'nlp_solver_type': 'SQP',
        # 'hessian_approx': 'GAUSS_NEWTON',
        'hessian_approx': 'EXACT',
        'integrator_type': 'IRK',
        'cost_type': 'EXTERNAL'
        # 'cost_type': 'LINEAR_LS'
    }
    set_acados_mpc(mpc)
    u_acados = mpc.make_step(x0)
    np.testing.assert_allclose(u_acados, u_ipopt)


if __name__ == '__main__':
    run_pendulum_mpc_without_array()