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

"""Tests for optimal control with multiple PDE constraints.

"""

from collections import namedtuple
import pathlib

from fenics import *
import pytest

import cashocs


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
def z(CG1):
    return Function(CG1)


@pytest.fixture
def p(CG1):
    return Function(CG1)


@pytest.fixture
def q(CG1):
    return Function(CG1)


@pytest.fixture
def u(CG1):
    return Function(CG1)


@pytest.fixture
def v(CG1):
    return Function(CG1)


@pytest.fixture
def states(y, z):
    return [y, z]


@pytest.fixture
def adjoints(p, q):
    return [p, q]


@pytest.fixture
def controls(u, v):
    return [u, v]


@pytest.fixture
def state_form_y(y, p, u, geometry):
    return dot(grad(y), grad(p)) * geometry.dx - u * p * geometry.dx


@pytest.fixture
def state_form_z(z, q, y, v, geometry):
    return dot(grad(z), grad(q)) * geometry.dx - (y + v) * q * geometry.dx


@pytest.fixture
def state_forms(state_form_y, state_form_z):
    return [state_form_y, state_form_z]


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
def bcs_list(bcs1, bcs2):
    return [bcs1, bcs2]


@pytest.fixture
def y_d():
    return Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)


@pytest.fixture
def z_d():
    return Expression("sin(4*pi*x[0])*sin(4*pi*x[1])", degree=1)


@pytest.fixture
def J(y, y_d, z, z_d, u, v, geometry):
    alpha = 1e-6
    beta = 1e-4
    return cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5) * (z - z_d) * (z - z_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
        + Constant(0.5 * beta) * v * v * geometry.dx
    )


@pytest.fixture
def ocp(state_forms, bcs_list, J, states, controls, adjoints, config_ocp):
    return cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )


def test_control_gradient_multiple(ocp, rng):
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_control_gd_multiple(ocp):
    ocp.solve(algorithm="gd", rtol=1e-2, atol=0.0, max_iter=47)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_fr_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoCG", "cg_method", "FR")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=21)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_pr_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoCG", "cg_method", "PR")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=36)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_hs_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoCG", "cg_method", "HS")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=30)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_dy_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoCG", "cg_method", "DY")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=13)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_hz_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoCG", "cg_method", "HZ")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=26)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_bfgs_multiple(ocp):
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=11)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_newton_cg_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoTNM", "inner_newton", "cg")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="newton", rtol=1e-2, atol=0.0, max_iter=2)
    assert ocp.solver.relative_norm <= 1e-4


def test_control_newton_cr_multiple(
    state_forms, bcs_list, J, states, controls, adjoints, config_ocp
):
    config_ocp.set("AlgoTNM", "inner_newton", "cr")
    ocp = cashocs.OptimalControlProblem(
        state_forms, bcs_list, J, states, controls, adjoints, config=config_ocp
    )
    ocp.solve(algorithm="newton", rtol=1e-2, atol=0.0, max_iter=2)
    assert ocp.solver.relative_norm <= 1e-4
