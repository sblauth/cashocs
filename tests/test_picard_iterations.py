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

from collections import namedtuple
import pathlib

from fenics import *
import numpy as np
import pytest

import cashocs

set_log_level(LogLevel.CRITICAL)


@pytest.fixture
def config_picard(dir_path):
    return cashocs.load_config(dir_path + "/config_picard.ini")


@pytest.fixture
def geometry():
    Geometry = namedtuple("Geometry", "mesh boundaries dx ds")
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(10)
    geom = Geometry(mesh, boundaries, dx, ds)

    return geom


@pytest.fixture
def CG1(geometry):
    return FunctionSpace(geometry.mesh, "CG", 1)


@pytest.fixture
def y(CG1):
    return Function(CG1)


@pytest.fixture
def p(CG1):
    return Function(CG1)


@pytest.fixture
def z(CG1):
    return Function(CG1)


@pytest.fixture
def q(CG1):
    return Function(CG1)


@pytest.fixture
def states(y, z):
    return [y, z]


@pytest.fixture
def adjoints(p, q):
    return [p, q]


@pytest.fixture
def u(CG1):
    return Function(CG1)


@pytest.fixture
def v(CG1):
    return Function(CG1)


@pytest.fixture
def controls(u, v):
    return [u, v]


@pytest.fixture
def e1(y, p, u, z, geometry):
    return (
        dot(grad(y), grad(p)) * geometry.dx + z * p * geometry.dx - u * p * geometry.dx
    )


@pytest.fixture
def e2(z, q, y, v, geometry):
    return (
        inner(grad(z), grad(q)) * geometry.dx
        + y * q * geometry.dx
        - v * q * geometry.dx
    )


@pytest.fixture
def e(e1, e2):
    return [e1, e2]


@pytest.fixture
def e1_nonlinear(y, p, z, u, geometry):
    return (
        inner(grad(y), grad(p)) * geometry.dx
        + pow(y, 3) * p * geometry.dx
        + z * p * geometry.dx
        - u * p * geometry.dx
    )


@pytest.fixture
def e2_nonlinear(z, q, y, v, geometry):
    return (
        inner(grad(z), grad(q)) * geometry.dx
        + pow(z, 3) * q * geometry.dx
        + y * q * geometry.dx
        - v * q * geometry.dx
    )


@pytest.fixture
def e_nonlinear(e1_nonlinear, e2_nonlinear):
    return [e1_nonlinear, e2_nonlinear]


@pytest.fixture
def bcs1(CG1, geometry):
    return cashocs.create_dirichlet_bcs(
        CG1, Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )


@pytest.fixture
def bcs2(CG1, geometry):
    return cashocs.create_dirichlet_bcs(
        CG1, Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )


@pytest.fixture
def bcs(bcs1, bcs2):
    return [bcs1, bcs2]


@pytest.fixture
def y_d():
    return Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)


@pytest.fixture
def z_d():
    return Expression("sin(4*pi*x[0])*sin(4*pi*x[1])", degree=1)


@pytest.fixture
def alpha():
    return 1e-4


@pytest.fixture
def beta():
    return 1e-4


@pytest.fixture
def J(y, z, u, v, geometry, alpha, beta, y_d, z_d):

    return cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5) * (z - z_d) * (z - z_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
        + Constant(0.5 * beta) * v * v * geometry.dx
    )


@pytest.fixture
def ocp(e, bcs, J, states, controls, adjoints, config_picard):
    return cashocs.OptimalControlProblem(
        e, bcs, J, states, controls, adjoints, config=config_picard
    )


@pytest.fixture
def Mixed(geometry):
    elem = FiniteElement("CG", geometry.mesh.ufl_cell(), 1)
    return FunctionSpace(geometry.mesh, MixedElement([elem, elem]))


@pytest.fixture
def state_m(Mixed):
    return Function(Mixed)


@pytest.fixture
def adjoint_m(Mixed):
    return Function(Mixed)


@pytest.fixture
def F(state_m, adjoint_m, u, v, geometry):
    y_m, z_m = split(state_m)
    p_m, q_m = split(adjoint_m)
    return (
        inner(grad(y_m), grad(p_m)) * geometry.dx
        + z_m * p_m * geometry.dx
        - u * p_m * geometry.dx
        + inner(grad(z_m), grad(q_m)) * geometry.dx
        + y_m * q_m * geometry.dx
        - v * q_m * geometry.dx
    )


@pytest.fixture
def bcs_m(Mixed, geometry):
    bcs_m1 = cashocs.create_dirichlet_bcs(
        Mixed.sub(0), Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )
    bcs_m2 = cashocs.create_dirichlet_bcs(
        Mixed.sub(1), Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )
    return bcs_m1 + bcs_m2


@pytest.fixture
def J_m(state_m, y_d, z_d, u, v, geometry, alpha, beta):
    y_m, z_m = split(state_m)
    return cashocs.IntegralFunctional(
        Constant(0.5) * (y_m - y_d) * (y_m - y_d) * geometry.dx
        + Constant(0.5) * (z_m - z_d) * (z_m - z_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
        + Constant(0.5 * beta) * v * v * geometry.dx
    )


@pytest.fixture
def ocp_mixed(F, bcs_m, J_m, state_m, controls, adjoint_m, config_picard):
    return cashocs.OptimalControlProblem(
        F, bcs_m, J_m, state_m, controls, adjoint_m, config=config_picard
    )


@pytest.fixture
def state_newton(Mixed):
    return Function(Mixed)


@pytest.fixture
def y_newton(state_newton):
    return split(state_newton)[0]


@pytest.fixture
def z_newton(state_newton):
    return split(state_newton)[1]


@pytest.fixture
def p_newton(Mixed):
    return TestFunctions(Mixed)[0]


@pytest.fixture
def q_newton(Mixed):
    return TestFunctions(Mixed)[1]


@pytest.fixture
def F_newton(y_newton, p_newton, z_newton, u, q_newton, v, geometry):
    return (
        inner(grad(y_newton), grad(p_newton)) * geometry.dx
        + z_newton * p_newton * geometry.dx
        - u * p_newton * geometry.dx
        + inner(grad(z_newton), grad(q_newton)) * geometry.dx
        + y_newton * q_newton * geometry.dx
        - v * q_newton * geometry.dx
    )


def test_picard_gradient_computation(ocp, rng):
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_picard_state_solver(
    rng, u, v, ocp, ocp_mixed, state_m, F_newton, state_newton, bcs_m, y, z
):
    u.vector().set_local(rng.normal(0.0, 10.0, size=u.vector().local_size()))
    u.vector().apply("")
    v.vector().set_local(rng.normal(0.0, 10.0, size=v.vector().local_size()))
    v.vector().apply("")
    ocp.compute_state_variables()
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


def test_picard_solver_for_optimization(ocp, ocp_mixed, u, v, CG1):
    u_picard = Function(CG1)
    v_picard = Function(CG1)

    ocp.solve(algorithm="newton", rtol=1e-6, atol=0.0, max_iter=2)
    assert ocp.solver.relative_norm <= 1e-6

    u_picard.vector()[:] = u.vector()[:]
    v_picard.vector()[:] = v.vector()[:]

    u.vector().vec().set(0.0)
    u.vector().apply("")
    v.vector().vec().set(0.0)
    v.vector().apply("")
    ocp_mixed.solve(algorithm="newton", rtol=1e-6, atol=0.0, max_iter=2)
    assert ocp_mixed.solver.relative_norm < 1e-6

    assert np.allclose(u.vector()[:], u_picard.vector()[:])
    assert (
        np.max(np.abs(u.vector()[:] - u_picard.vector()[:]))
        / np.max(np.abs(u.vector()[:]))
        <= 1e-8
    )


def test_picard_nonlinear(
    e_nonlinear, bcs, J, states, controls, adjoints, config_picard, rng
):
    config_picard.set("StateSystem", "is_linear", "False")
    config_picard.set("OptimizationRoutine", "algorithm", "newton")
    config_picard.set("OptimizationRoutine", "rtol", "1e-6")
    config_picard.set("OptimizationRoutine", "atol", "0.0")
    config_picard.set("OptimizationRoutine", "maximum_iterations", "10")

    ocp_nonlinear = cashocs.OptimalControlProblem(
        e_nonlinear, bcs, J, states, controls, adjoints, config=config_picard
    )

    assert ocp_nonlinear.gradient_test(rng=rng) > 1.9
    assert ocp_nonlinear.gradient_test(rng=rng) > 1.9
    assert ocp_nonlinear.gradient_test(rng=rng) > 1.9

    ocp_nonlinear.solve(algorithm="newton", rtol=1e-6, atol=0.0, max_iter=10)
    assert ocp_nonlinear.solver.relative_norm < 1e-6
