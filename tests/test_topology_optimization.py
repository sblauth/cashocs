# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

from collections import namedtuple

from fenics import *
import numpy as np
import pytest

import cashocs
from cashocs._optimization.topology_optimization import bisection

rng = np.random.RandomState(300696)


@pytest.fixture
def geometry():
    Geometry = namedtuple("Geometry", "mesh subdomains boundaries dx ds dS")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(
        16, length_x=2.0, diagonal="crossed"
    )
    geom = Geometry(mesh, subdomains, boundaries, dx, ds, dS)

    return geom


@pytest.fixture
def CG1(geometry):
    return FunctionSpace(geometry.mesh, "CG", 1)


@pytest.fixture
def DG0(geometry):
    return FunctionSpace(geometry.mesh, "DG", 0)


@pytest.fixture
def VCG1(geometry):
    return VectorFunctionSpace(geometry.mesh, "CG", 1)


@pytest.fixture
def u(VCG1):
    return Function(VCG1)


@pytest.fixture
def v(VCG1):
    return Function(VCG1)


@pytest.fixture
def psi(CG1):
    levelset = Function(CG1)
    levelset.vector()[:] = -1.0
    return levelset


@pytest.fixture
def psi_proj(CG1):
    levelset = Function(CG1)
    levelset.vector()[:] = -1.0
    return levelset


@pytest.fixture
def alpha(DG0):
    return Function(DG0)


@pytest.fixture
def indicator_omega(DG0):
    return Function(DG0)


@pytest.fixture
def mu():
    E = 1.0
    nu = 0.3
    return E / (2.0 * (1.0 + nu))


@pytest.fixture
def lambd(mu):
    E = 1.0
    nu = 0.3
    plane_stress = True

    lambd_eval = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if plane_stress:
        lambd_eval = 2 * mu * lambd_eval / (lambd_eval + 2.0 * mu)
    return lambd_eval


@pytest.fixture
def kappa(lambd, mu):
    return (lambd + 3.0 * mu) / (lambd + mu)


@pytest.fixture
def alpha_in():
    return 1.0


@pytest.fixture
def alpha_out():
    return 1e-3


@pytest.fixture
def r_in(alpha_in, alpha_out):
    return alpha_out / alpha_in


@pytest.fixture
def r_out(alpha_in, alpha_out):
    return alpha_in / alpha_out


@pytest.fixture
def gamma():
    return 100.0


@pytest.fixture
def eps():
    def eps_eval(u):
        return Constant(0.5) * (grad(u) + grad(u).T)

    return eps_eval


@pytest.fixture
def sigma(mu, lambd, eps):
    def sigma_eval(u):
        return Constant(2.0 * mu) * eps(u) + Constant(lambd) * tr(eps(u)) * Identity(2)

    return sigma_eval


@pytest.fixture
def F(alpha, u, v, geometry, eps, sigma):
    class Delta(UserExpression):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def eval(self, values, x):
            if near(x[0], 2.0) and near(x[1], 0.5):
                values[0] = 3.0 / geometry.mesh.hmax()
            else:
                values[0] = 0.0

        def value_shape(self):
            return ()

    delta = Delta(degree=2)
    g = delta * Constant((0.0, -1.0))

    return alpha * inner(sigma(u), eps(v)) * geometry.dx - dot(g, v) * geometry.ds(2)


@pytest.fixture
def bcs(VCG1, geometry):
    return cashocs.create_dirichlet_bcs(
        VCG1, Constant((0.0, 0.0)), geometry.boundaries, 1
    )


@pytest.fixture
def J(u, alpha, indicator_omega, geometry, eps, sigma, gamma):
    return cashocs.IntegralFunctional(
        alpha * inner(sigma(u), eps(u)) * geometry.dx
        + Constant(gamma) * indicator_omega * geometry.dx
    )


@pytest.fixture
def J_proj(u, alpha, indicator_omega, geometry, eps, sigma):
    return cashocs.IntegralFunctional(alpha * inner(sigma(u), eps(u)) * geometry.dx)


@pytest.fixture
def dJ_in(u, eps, sigma, kappa, alpha_in, alpha_out, r_in, r_out, gamma):
    dJ = (
        Constant(alpha_in * (r_in - 1.0) / (kappa * r_in + 1.0) * (kappa + 1.0) / 2.0)
        * (
            Constant(2.0) * inner(sigma(u), eps(u))
            + Constant((r_in - 1.0) * (kappa - 2.0) / (kappa + 2 * r_in - 1.0))
            * tr(sigma(u))
            * tr(eps(u))
        )
    ) + Constant(gamma)

    return dJ


@pytest.fixture
def dJ_out(u, eps, sigma, kappa, alpha_in, alpha_out, r_in, r_out, gamma):
    dJ = (
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

    return dJ


@pytest.fixture
def dJ_in_proj(u, eps, sigma, kappa, alpha_in, alpha_out, r_in, r_out):
    dJ = Constant(
        alpha_in * (r_in - 1.0) / (kappa * r_in + 1.0) * (kappa + 1.0) / 2.0
    ) * (
        Constant(2.0) * inner(sigma(u), eps(u))
        + Constant((r_in - 1.0) * (kappa - 2.0) / (kappa + 2 * r_in - 1.0))
        * tr(sigma(u))
        * tr(eps(u))
    )

    return dJ


@pytest.fixture
def dJ_out_proj(u, eps, sigma, kappa, alpha_in, alpha_out, r_in, r_out):
    dJ = Constant(
        -alpha_out * (r_out - 1.0) / (kappa * r_out + 1.0) * (kappa + 1.0) / 2.0
    ) * (
        Constant(2.0) * inner(sigma(u), eps(u))
        + Constant((r_out - 1.0) * (kappa - 2.0) / (kappa + 2 * r_out - 1.0))
        * tr(sigma(u))
        * tr(eps(u))
    )

    return dJ


@pytest.fixture
def update_level_set(psi, alpha, indicator_omega, alpha_in, alpha_out):
    def update_level_set_eval():
        cashocs.interpolate_levelset_function_to_cells(psi, alpha_in, alpha_out, alpha)
        cashocs.interpolate_levelset_function_to_cells(psi, 1.0, 0.0, indicator_omega)

    return update_level_set_eval


@pytest.fixture
def update_level_set_proj(psi, alpha, alpha_in, alpha_out):
    def update_level_set_eval():
        cashocs.interpolate_levelset_function_to_cells(psi, alpha_in, alpha_out, alpha)

    return update_level_set_eval


@pytest.fixture
def levelset_function(CG1, geometry):
    psi_exp = Expression("x[0]-0.5", degree=1)
    psi = Function(CG1)
    psi.vector()[:] = project(psi_exp, CG1).vector()[:]

    return psi


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
    F,
    bcs,
    J,
    u,
    v,
    psi,
    dJ_in,
    dJ_out,
    update_level_set,
    config_top,
    algorithm,
    iter,
    tol,
):
    top = cashocs.TopologyOptimizationProblem(
        F, bcs, J, u, v, psi, dJ_in, dJ_out, update_level_set, config=config_top
    )
    top.solve(algorithm=algorithm, rtol=0.0, atol=0.0, angle_tol=tol, max_iter=iter)


def test_evaluate_volume(
    F,
    bcs,
    J,
    u,
    v,
    levelset_function,
    dJ_in,
    update_level_set,
    config_top,
):
    top = cashocs.TopologyOptimizationProblem(
        F,
        bcs,
        J,
        u,
        v,
        levelset_function,
        dJ_in,
        dJ_in,
        update_level_set,
        config=config_top,
    )
    vol = top.projection.vol
    eval_vol = top.projection.evaluate(0.0, 0.0)
    assert abs(eval_vol - vol) < top.projection.tol_bisect


@pytest.mark.parametrize(
    "algorithm,vol",
    [
        ("sphere_combination", [rng.rand(), rng.rand() + 1.0]),
        ("convex_combination", [rng.rand(), rng.rand() + 1.0]),
        ("gradient_descent", [rng.rand(), rng.rand() + 1.0]),
        ("bfgs", [rng.rand(), rng.rand() + 1.0]),
        ("sphere_combination", 2 * rng.rand()),
        ("convex_combination", 2 * rng.rand()),
        ("gradient_descent", 2 * rng.rand()),
        ("bfgs", 2 * rng.rand()),
    ],
)
def test_topology_optimization_algorithms_for_cantilever_projection(
    F,
    bcs,
    J_proj,
    u,
    v,
    psi_proj,
    dJ_in_proj,
    dJ_out_proj,
    update_level_set_proj,
    config_top,
    algorithm,
    vol,
):
    config_top.set("OptimizationRoutine", "soft_exit", "True")

    top = cashocs.TopologyOptimizationProblem(
        F,
        bcs,
        J_proj,
        u,
        v,
        psi_proj,
        dJ_in_proj,
        dJ_out_proj,
        update_level_set_proj,
        config=config_top,
        volume_restriction=vol,
    )
    top.solve(algorithm=algorithm, max_iter=2)
    volume = top.projection.evaluate(0.0, 0.0)
    if isinstance(vol, float):
        assert abs(volume - vol) < top.projection.tol_bisect
    else:
        res = (volume + top.projection.tol_bisect - vol[0]) * (
            volume - top.projection.tol_bisect - vol[1]
        )
        assert res < 0


@pytest.mark.parametrize(
    "volume",
    [
        ([rng.rand() / 4.0 + 0.5, rng.rand() / 4.0 + 0.75]),
        ([rng.rand() / 4.0, rng.rand() / 4.0 + 0.25]),
        ([rng.rand() / 2.0, rng.rand() / 2.0 + 0.5]),
        (rng.rand()),
    ],
)
def test_projection_method_for_topology_optimization(
    F, bcs, J, u, v, levelset_function, dJ_in, update_level_set, volume, config_top
):
    if isinstance(volume, float):
        target = volume
    else:
        target = min(max(volume[0], 0.5), volume[1])

    top = cashocs.TopologyOptimizationProblem(
        F,
        bcs,
        J,
        u,
        v,
        levelset_function,
        dJ_in,
        dJ_in,
        update_level_set,
        volume_restriction=volume,
        config=config_top,
    )
    top.projection.project()
    projection_volume = top.projection.evaluate(0.0, 0.0)
    assert abs(projection_volume - target) < top.projection.tol_bisect
