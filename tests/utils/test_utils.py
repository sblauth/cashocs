# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
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

"""Tests for the utils module."""

import pathlib

from fenics import *
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs._exceptions import InputError
import cashocs._utils
import cashocs._utils.linalg

rng = np.random.RandomState(300696)
mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)


@pytest.fixture
def ocp(config_ocp):
    return cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config_ocp)


def test_summation():
    a = [1, 2, 3, 4]

    dim = 3
    funcs = []
    test = TestFunction(V)

    for i in range(dim):
        temp = Function(V)
        temp.vector().set_local(rng.rand(temp.vector().local_size()))
        temp.vector().apply("")
        funcs.append(temp)

    F = cashocs._utils.summation([funcs[i] * test * dx for i in range(dim)])
    F_exact = (funcs[0] + funcs[1] + funcs[2]) * test * dx

    b = assemble(F)[:]
    b_exact = assemble(F_exact)[:]

    assert cashocs._utils.summation(a) == 10
    assert abs(cashocs._utils.summation(b) - np.sum(b)) < 1e-14
    assert np.allclose(b, b_exact)
    assert assemble(pow(cashocs._utils.summation([]), 2) * dx) == 0.0


def test_multiplication():
    a = [1, 2, 3, 4]

    dim = 3
    funcs = []
    test = TestFunction(V)

    for i in range(dim):
        temp = Function(V)
        temp.vector().set_local(rng.rand(temp.vector().local_size()))
        temp.vector().apply("")
        funcs.append(temp)

    F = cashocs._utils.multiplication([funcs[i] for i in range(dim)]) * test * dx
    F_exact = (funcs[0] * funcs[1] * funcs[2]) * test * dx

    b = assemble(F)[:]
    b_exact = assemble(F_exact)[:]

    assert cashocs._utils.multiplication(a) == 24
    assert abs(cashocs._utils.multiplication(b) - np.prod(b)) < 1e-14
    assert np.allclose(b, b_exact)
    assert assemble(cashocs._utils.multiplication([]) * dx) / assemble(1 * dx) == 1.0


def test_interpolator():
    W = FunctionSpace(mesh, "CG", 2)
    X = FunctionSpace(mesh, "DG", 0)

    interp_W = cashocs.Interpolator(V, W)
    interp_X = cashocs.Interpolator(V, X)

    expr = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
    func_V = interpolate(expr, V)

    fen_W = interpolate(func_V, W)
    fen_X = interpolate(func_V, X)

    cas_W = interp_W.interpolate(func_V)
    cas_X = interp_X.interpolate(func_V)

    assert np.allclose(fen_W.vector()[:], cas_W.vector()[:])
    assert np.allclose(fen_X.vector()[:], cas_X.vector()[:])


def test_moreau_yosida_regularization(ocp):
    u.vector().vec().set(1e3)
    u.vector().apply("")
    y_bar = 1e-1
    y_low = 1e-2
    gamma = 1e3
    reg = cashocs._utils.moreau_yosida_regularization(
        y, gamma, dx, upper_threshold=y_bar, lower_threshold=y_low
    )

    max = cashocs._utils.max_
    min = cashocs._utils.min_
    reg_ana = (
        1 / (2 * gamma) * pow(max(gamma * (y - y_bar), 0.0), 2) * dx
        + 1 / (2 * gamma) * pow(min(gamma * (y - y_low), 0.0), 2) * dx
    )

    ocp.compute_state_variables()
    assert np.abs(assemble(reg) - assemble(reg_ana)) < 1e-14


def test_l2_projection():
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(8)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    expr = x[0] ** 2 + x[1]

    fenics_result = project(expr, V)
    cashocs_result = cashocs._utils.l2_projection(expr, V)

    assert (
        np.max(np.abs(fenics_result.vector()[:] - cashocs_result.vector()[:])) < 1e-10
    )
