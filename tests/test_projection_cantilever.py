import random

from fenics import *
import numpy as np
import pytest

import cashocs
from cashocs._optimization.topology_optimization import bisection

rng = np.random.RandomState(300696)


@pytest.fixture
def cantilever_problem(config_top):
    E = 1.0
    nu = 0.3
    plane_stress = True

    config_top.set("OptimizationRoutine", "soft_exit", "True")

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

    psi = Function(CG1)
    psi.vector()[:] = 1.0

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

    J = cashocs.IntegralFunctional(alpha * inner(sigma(u), eps(u)) * dx)

    kappa = (lambd + 3.0 * mu) / (lambd + mu)
    r_in = alpha_out / alpha_in
    r_out = alpha_in / alpha_out

    dJ_in = Constant(
        alpha_in * (r_in - 1.0) / (kappa * r_in + 1.0) * (kappa + 1.0) / 2.0
    ) * (
        Constant(2.0) * inner(sigma(u), eps(u))
        + Constant((r_in - 1.0) * (kappa - 2.0) / (kappa + 2 * r_in - 1.0))
        * tr(sigma(u))
        * tr(eps(u))
    )
    dJ_out = Constant(
        -alpha_out * (r_out - 1.0) / (kappa * r_out + 1.0) * (kappa + 1.0) / 2.0
    ) * (
        Constant(2.0) * inner(sigma(u), eps(u))
        + Constant((r_out - 1.0) * (kappa - 2.0) / (kappa + 2 * r_out - 1.0))
        * tr(sigma(u))
        * tr(eps(u))
    )

    def update_level_set():
        cashocs.interpolate_levelset_function_to_cells(psi, alpha_in, alpha_out, alpha)

    top = cashocs.TopologyOptimizationProblem(
        F, bcs, J, u, v, psi, dJ_in, dJ_out, update_level_set, config=config_top
    )

    return top


@pytest.mark.parametrize(
    "algorithm,vol",
    [
        ("sphere_combination", [rng.rand(), rng.rand() + 1.0]),
        ("convex_combination", [rng.rand(), rng.rand() + 1.0]),
        ("gradient_descent", [rng.rand(), rng.rand() + 1.0]),
        ("bfgs", [rng.rand(), rng.rand() + 1.0]),
        ("sphere_combination", [2 * rng.rand()]),
        ("convex_combination", [2 * rng.rand()]),
        ("gradient_descent", [2 * rng.rand()]),
        ("bfgs", [2 * rng.rand()]),
    ],
)
def test_topology_optimization_algorithms_for_cantilever(
    cantilever_problem, algorithm, vol
):
    cantilever_problem.projection = bisection.projection_levelset(
        cantilever_problem.levelset_function, volume_restriction=vol
    )
    cantilever_problem.solve(algorithm=algorithm, max_iter=2)
    volume = cantilever_problem.projection.evaluate(0.0, 0.0)
    if len(vol) == 1:
        assert abs(volume - vol[0]) < cantilever_problem.projection.tolerance_bisect
    else:
        res = (volume + cantilever_problem.projection.tolerance_bisect - vol[0]) * (
            volume - cantilever_problem.projection.tolerance_bisect - vol[1]
        )
        assert res < 0
