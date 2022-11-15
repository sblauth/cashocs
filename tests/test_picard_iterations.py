# Copyright (C) 2020-2022 Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for the Picard iteration.

"""

import pathlib

from fenics import *
import numpy as np

import cashocs

rng = np.random.RandomState(300696)
set_log_level(LogLevel.CRITICAL)
dir_path = str(pathlib.Path(__file__).parent)
config = cashocs.load_config(dir_path + "/config_picard.ini")

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
z = Function(V)
p = Function(V)
q = Function(V)
states = [y, z]
adjoints = [p, q]

u = Function(V)
v = Function(V)
controls = [u, v]

e1 = inner(grad(y), grad(p)) * dx + z * p * dx - u * p * dx
e2 = inner(grad(z), grad(q)) * dx + y * q * dx - v * q * dx
e = [e1, e2]

e1_nonlinear = (
    inner(grad(y), grad(p)) * dx + pow(y, 3) * p * dx + z * p * dx - u * p * dx
)
e2_nonlinear = (
    inner(grad(z), grad(q)) * dx + pow(z, 3) * q * dx + y * q * dx - v * q * dx
)
e_nonlinear = [e1_nonlinear, e2_nonlinear]

bcs1 = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs2 = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs = [bcs1, bcs2]

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
z_d = Expression("sin(4*pi*x[0])*sin(4*pi*x[1])", degree=1)
alpha = 1e-4
beta = 1e-4
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx
    + Constant(0.5) * (z - z_d) * (z - z_d) * dx
    + Constant(0.5 * alpha) * u * u * dx
    + Constant(0.5 * beta) * v * v * dx
)

ocp = cashocs.OptimalControlProblem(e, bcs, J, states, controls, adjoints, config)


elem = FiniteElement("CG", mesh.ufl_cell(), 1)
Mixed = FunctionSpace(mesh, MixedElement([elem, elem]))

state_m = Function(Mixed)
y_m, z_m = split(state_m)
adjoint_m = Function(Mixed)
p_m, q_m = split(adjoint_m)

F = (
    inner(grad(y_m), grad(p_m)) * dx
    + z_m * p_m * dx
    - u * p_m * dx
    + inner(grad(z_m), grad(q_m)) * dx
    + y_m * q_m * dx
    - v * q_m * dx
)
bcs_m1 = cashocs.create_dirichlet_bcs(
    Mixed.sub(0), Constant(0), boundaries, [1, 2, 3, 4]
)
bcs_m2 = cashocs.create_dirichlet_bcs(
    Mixed.sub(1), Constant(0), boundaries, [1, 2, 3, 4]
)
bcs_m = bcs_m1 + bcs_m2

J_m = cashocs.IntegralFunctional(
    Constant(0.5) * (y_m - y_d) * (y_m - y_d) * dx
    + Constant(0.5) * (z_m - z_d) * (z_m - z_d) * dx
    + Constant(0.5 * alpha) * u * u * dx
    + Constant(0.5 * beta) * v * v * dx
)

ocp_mixed = cashocs.OptimalControlProblem(
    F, bcs_m, J_m, state_m, controls, adjoint_m, config
)


state_newton = Function(Mixed)
y_newton, z_newton = split(state_newton)
p_newton, q_newton = TestFunctions(Mixed)

F_newton = (
    inner(grad(y_newton), grad(p_newton)) * dx
    + z_newton * p_newton * dx
    - u * p_newton * dx
    + inner(grad(z_newton), grad(q_newton)) * dx
    + y_newton * q_newton * dx
    - v * q_newton * dx
)


def test_picard_gradient_computation():
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_picard_state_solver():
    u.vector().set_local(rng.normal(0.0, 10.0, size=u.vector().local_size()))
    u.vector().apply("")
    v.vector().set_local(rng.normal(0.0, 10.0, size=v.vector().local_size()))
    v.vector().apply("")
    ocp._erase_pde_memory()
    ocp.compute_state_variables()
    ocp_mixed._erase_pde_memory()
    ocp_mixed.compute_state_variables()
    y_m, z_m = state_m.split(True)

    solve(F_newton == 0, state_newton, bcs_m)
    y_ref, z_ref = state_newton.split(True)

    assert np.allclose(y.vector()[:], y_ref.vector()[:])
    assert np.allclose(y_m.vector()[:], y_ref.vector()[:])
    assert (
        np.max(np.abs(y.vector()[:] - y_ref.vector()[:]))
        / np.max(np.abs(y_ref.vector()[:]))
        <= 1e-13
    )
    assert (
        np.max(np.abs(y_m.vector()[:] - y_ref.vector()[:]))
        / np.max(np.abs(y_ref.vector()[:]))
        <= 1e-13
    )

    assert np.allclose(z.vector()[:], z_ref.vector()[:])
    assert np.allclose(z_m.vector()[:], z_ref.vector()[:])
    assert (
        np.max(np.abs(z.vector()[:] - z_ref.vector()[:]))
        / np.max(np.abs(z_ref.vector()[:]))
        <= 1e-13
    )
    assert (
        np.max(np.abs(z_m.vector()[:] - z_ref.vector()[:]))
        / np.max(np.abs(z_ref.vector()[:]))
        <= 1e-13
    )


def test_picard_solver_for_optimization():
    config = cashocs.load_config(dir_path + "/config_picard.ini")
    config.set("OptimizationRoutine", "algorithm", "newton")
    config.set("OptimizationRoutine", "rtol", "1e-6")
    config.set("OptimizationRoutine", "atol", "0.0")
    config.set("OptimizationRoutine", "maximum_iterations", "2")

    u_picard = Function(V)
    v_picard = Function(V)

    u.vector().vec().set(0.0)
    u.vector().apply("")
    v.vector().vec().set(0.0)
    v.vector().apply("")
    ocp = cashocs.OptimalControlProblem(e, bcs, J, states, controls, adjoints, config)
    ocp.solve()
    assert ocp.solver.relative_norm <= 1e-6

    u_picard.vector()[:] = u.vector()[:]
    v_picard.vector()[:] = v.vector()[:]

    u.vector().vec().set(0.0)
    u.vector().apply("")
    v.vector().vec().set(0.0)
    v.vector().apply("")
    ocp_mixed = cashocs.OptimalControlProblem(
        F, bcs_m, J_m, state_m, controls, adjoint_m, config
    )
    ocp_mixed.solve()
    assert ocp_mixed.solver.relative_norm < 1e-6

    assert np.allclose(u.vector()[:], u_picard.vector()[:])
    assert (
        np.max(np.abs(u.vector()[:] - u_picard.vector()[:]))
        / np.max(np.abs(u.vector()[:]))
        <= 1e-8
    )


def test_picard_nonlinear():
    config = cashocs.load_config(dir_path + "/config_picard.ini")

    u.vector().vec().set(0.0)
    u.vector().apply("")
    v.vector().vec().set(0.0)
    v.vector().apply("")
    config.set("StateSystem", "is_linear", "False")
    config.set("OptimizationRoutine", "algorithm", "newton")
    config.set("OptimizationRoutine", "rtol", "1e-6")
    config.set("OptimizationRoutine", "atol", "0.0")
    config.set("OptimizationRoutine", "maximum_iterations", "10")

    ocp_nonlinear = cashocs.OptimalControlProblem(
        e_nonlinear, bcs, J, states, controls, adjoints, config
    )

    assert ocp_nonlinear.gradient_test(rng=rng) > 1.9
    assert ocp_nonlinear.gradient_test(rng=rng) > 1.9
    assert ocp_nonlinear.gradient_test(rng=rng) > 1.9

    ocp_nonlinear.solve()
    assert ocp_nonlinear.solver.relative_norm < 1e-6
