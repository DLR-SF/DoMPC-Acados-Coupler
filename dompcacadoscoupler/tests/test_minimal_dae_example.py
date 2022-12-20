import casadi as cd
import numpy as np
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver


def test_minimal_dae_example():
    model_name = 'minimal_example'
    # set up states & controls
    x = cd.SX.sym('x')
    u = cd.SX.sym('u')
    # xdot
    x_dot = cd.SX.sym('x_dot')
    # algebraic variables
    z = cd.SX.sym('z')
    # dynamics
    f_expl = 1 + z
    f_impl = x_dot - f_expl
    alg = z - 1
    f_impl = cd.vertcat(f_impl, alg)

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.z = z
    # set model_name
    model.name = model_name
    sim = AcadosSim()
    sim.model = model

    Tf = 1
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    # set simulation time
    sim.solver_options.T = Tf
    # set options
    sim.solver_options.integrator_type = 'IRK'

    sim.solver_options.output_z = True

    # create
    acados_integrator = AcadosSimSolver(sim)

    x0 = np.array([0.0])
    u0 = np.array([0.0])
    acados_integrator.set("u", u0)

    # set initial state
    acados_integrator.set("x", x0)
    # initialize IRK
    acados_integrator.set("xdot", np.zeros((nx,)))

    # solve
    status = acados_integrator.solve()
    if status != 0:
        raise Exception(f'acados returned status {status}.')
    # get solution
    x_new = acados_integrator.get("x")
    z_new = acados_integrator.get("z")
    np.testing.assert_allclose(x_new, 2)  # works
    np.testing.assert_allclose(z_new, 1)  # fails


if __name__ == "__main__":
    test_minimal_dae_example()
