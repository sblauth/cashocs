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

"""Tests for the module PDE problems.

"""

from collections import namedtuple

from fenics import *
import numpy as np
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
def p(CG1):
    return Function(CG1)


@pytest.fixture
def u(CG1):
    return Function(CG1)


@pytest.fixture
def e(y, u, p, geometry):
    return dot(grad(y), grad(p)) * geometry.dx - u * p * geometry.dx


@pytest.fixture
def bcs(CG1, geometry):
    return cashocs.create_dirichlet_bcs(CG1, Constant(0), geometry.boundaries, [1, 2])


@pytest.fixture
def y_d(CG1):
    return Function(CG1)


@pytest.fixture
def J(y, y_d, u, geometry):
    alpha = 1e-6
    return cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
    )


@pytest.fixture
def ocp(e, bcs, J, y, u, p, config_ocp):
    return cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config=config_ocp)


def test_state_adjoint_problems(CG1, geometry, rng, u, y_d, ocp, bcs, y, p):
    trial = TrialFunction(CG1)
    test = TestFunction(CG1)
    state = Function(CG1)
    adjoint = Function(CG1)

    a = inner(grad(trial), grad(test)) * geometry.dx
    L_state = u * test * geometry.dx
    L_adjoint = -(state - y_d) * test * geometry.dx

    y_d.vector().set_local(rng.rand(y_d.vector().local_size()))
    y_d.vector().apply("")
    u.vector().set_local(rng.rand(u.vector().local_size()))
    u.vector().apply("")

    ocp.compute_state_variables()
    ocp.compute_adjoint_variables()

    solve(a == L_state, state, bcs)
    solve(a == L_adjoint, adjoint, bcs)

    assert np.allclose(state.vector()[:], y.vector()[:])
    assert np.allclose(adjoint.vector()[:], p.vector()[:])


def test_control_gradient(CG1, geometry, rng, ocp, u, p, y_d):
    trial = TrialFunction(CG1)
    test = TestFunction(CG1)
    gradient = Function(CG1)

    a = trial * test * geometry.dx
    L = Constant(1e-6) * u * test * geometry.dx - p * test * geometry.dx

    y_d.vector().set_local(rng.rand(y_d.vector().local_size()))
    y_d.vector().apply("")
    u.vector().set_local(rng.rand(u.vector().local_size()))
    u.vector().apply("")

    c_gradient = ocp.compute_gradient()[0]
    solve(a == L, gradient)

    assert np.allclose(c_gradient.vector()[:], gradient.vector()[:])
