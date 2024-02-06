# Copyright (C) 2020-2024 Sebastian Blauth
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

"""Tests for topology optimization problems."""

from fenics import *
import numpy as np
import pytest

import cashocs


@pytest.fixture
def cantilever_problem(config_top):
    gamma = 100.0

    E = 1.0
    nu = 0.3
    plane_stress = True

    mu = E / (2.0 * (1.0 + nu))
    lambd = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if plane_stress:
        lambd = 2 * mu * lambd / (lambd + 2.0 * mu)

    alpha_in = 1.0
    alpha_out = 1e-3

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(
        16, length_x=2.0, diagonal="crossed"
    )
    V = VectorFunctionSpace(mesh, "CG", 1)
    CG1 = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)

    alpha = Function(DG0)
    indicator_omega = Function(DG0)

    psi = Function(CG1)
    psi.vector()[:] = -1.0

    def eps(u):
        return Constant(0.5) * (grad(u) + grad(u).T)

    def sigma(u):
        return Constant(2.0 * mu) * eps(u) + Constant(lambd) * tr(eps(u)) * Identity(2)

    class Delta(UserExpression):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def eval(self, values, x):
            if near(x[0], 2.0) and near(x[1], 0.5):
                values[0] = 3.0 / mesh.hmax()
            else:
                values[0] = 0.0

        def value_shape(self):
            return ()

    delta = Delta(degree=2)
    g = delta * Constant((0.0, -1.0))

    u = Function(V)
    v = Function(V)
    F = alpha * inner(sigma(u), eps(v)) * dx - dot(g, v) * ds(2)
    bcs = cashocs.create_dirichlet_bcs(V, Constant((0.0, 0.0)), boundaries, 1)

    J = cashocs.IntegralFunctional(
        alpha * inner(sigma(u), eps(u)) * dx + Constant(gamma) * indicator_omega * dx
    )

    kappa = (lambd + 3.0 * mu) / (lambd + mu)
    r_in = alpha_out / alpha_in
    r_out = alpha_in / alpha_out

    dJ_in = (
        Constant(alpha_in * (r_in - 1.0) / (kappa * r_in + 1.0) * (kappa + 1.0) / 2.0)
        * (
            Constant(2.0) * inner(sigma(u), eps(u))
            + Constant((r_in - 1.0) * (kappa - 2.0) / (kappa + 2 * r_in - 1.0))
            * tr(sigma(u))
            * tr(eps(u))
        )
    ) + Constant(gamma)
    dJ_out = (
        Constant(
            -alpha_out * (r_out - 1.0) / (kappa * r_out + 1.0) * (kappa + 1.0) / 2.0
        )
        * (
            Constant(2.0) * inner(sigma(u), eps(u))
            + Constant((r_out - 1.0) * (kappa - 2.0) / (kappa + 2 * r_out - 1.0))
            * tr(sigma(u))
            * tr(eps(u))
        )
    ) + Constant(gamma)

    def update_level_set():
        cashocs.interpolate_levelset_function_to_cells(psi, alpha_in, alpha_out, alpha)
        cashocs.interpolate_levelset_function_to_cells(psi, 1.0, 0.0, indicator_omega)

    psi.vector()[:] = -1.0
    top = cashocs.TopologyOptimizationProblem(
        F, bcs, J, u, v, psi, dJ_in, dJ_out, update_level_set, config=config_top
    )

    return top


@pytest.mark.parametrize(
    "algorithm,iter,tol",
    [
        ("sphere_combination", 16, 1.5),
        ("convex_combination", 13, 1.5),
        ("gradient_descent", 11, 3.5),
        ("bfgs", 19, 3.0),
    ],
)
def test_topology_optimization_algorithms_for_cantilever(
    cantilever_problem, algorithm, iter, tol
):
    cantilever_problem.solve(
        algorithm=algorithm, rtol=0.0, atol=0.0, angle_tol=tol, max_iter=iter
    )
