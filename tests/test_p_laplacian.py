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

from collections import namedtuple
import pathlib

from fenics import *
import numpy as np
import pytest

import cashocs


@pytest.fixture
def geometry(dir_path):
    Geometry = namedtuple("Geometry", "mesh boundaries dx ds")
    mesh, _, boundaries, dx, ds, _ = cashocs.import_mesh(
        f"{dir_path}/mesh/unit_circle/mesh.xdmf"
    )
    geom = Geometry(mesh, boundaries, dx, ds)

    return geom


@pytest.fixture
def CG1(geometry):
    return FunctionSpace(geometry.mesh, "CG", 1)


@pytest.fixture
def initial_coordinates(geometry):
    return geometry.mesh.coordinates().copy()


@pytest.fixture
def bcs(CG1, geometry):
    return DirichletBC(CG1, Constant(0), geometry.boundaries, 1)


@pytest.fixture
def f(geometry):
    x = SpatialCoordinate(geometry.mesh)
    return 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1


@pytest.fixture
def u(CG1):
    return Function(CG1)


@pytest.fixture
def p(CG1):
    return Function(CG1)


@pytest.fixture
def e(u, p, f, geometry):
    return dot(grad(u), grad(p)) * geometry.dx - f * p * geometry.dx


@pytest.fixture
def J(u, geometry):
    return cashocs.IntegralFunctional(u * geometry.dx)


def test_2_laplacian(config_sop, geometry, e, bcs, J, u, p, initial_coordinates):
    space = VectorFunctionSpace(geometry.mesh, "CG", 1)
    shape_scalar_product = (
        Constant(1)
        * inner((grad(TrialFunction(space))), (grad(TestFunction(space))))
        * geometry.dx
        + dot(TrialFunction(space), TestFunction(space)) * geometry.dx
    )

    config_sop.set("ShapeGradient", "mu_def", "1.0")
    config_sop.set("ShapeGradient", "mu_fix", "1.0")
    config_sop.set("ShapeGradient", "damping_factor", "1.0")
    config_sop.set("ShapeGradient", "use_p_laplacian", "True")
    config_sop.set("ShapeGradient", "p_laplacian_power", "2")
    config_sop.set("ShapeGradient", "p_laplacian_stabilization", "0.0")

    sop1 = cashocs.ShapeOptimizationProblem(
        e, bcs, J, u, p, geometry.boundaries, config=config_sop
    )
    sop1.solve(algorithm="gd", rtol=1e-2, max_iter=22)

    config_sop.set("ShapeGradient", "use_p_laplacian", "False")
    geometry.mesh.coordinates()[:, :] = initial_coordinates
    geometry.mesh.bounding_box_tree().build(geometry.mesh)
    sop2 = cashocs.ShapeOptimizationProblem(
        e,
        bcs,
        J,
        u,
        p,
        geometry.boundaries,
        config=config_sop,
        shape_scalar_product=shape_scalar_product,
    )
    sop2.solve(algorithm="gd", rtol=1e-2, max_iter=22)

    assert (
        np.abs(sop1.solver.objective_value - sop2.solver.objective_value)
        / np.abs(sop1.solver.objective_value)
        < 1e-10
    )
    assert (
        np.abs(sop1.solver.gradient_norm - sop2.solver.gradient_norm)
        / np.abs(sop1.solver.gradient_norm)
        < 1e-8
    )


def test_p_laplacian(config_sop, geometry, e, bcs, J, u, p):
    config_sop.set("ShapeGradient", "mu_def", "1.0")
    config_sop.set("ShapeGradient", "mu_fix", "1.0")
    config_sop.set("ShapeGradient", "damping_factor", "1.0")
    config_sop.set("ShapeGradient", "use_p_laplacian", "True")
    config_sop.set("ShapeGradient", "p_laplacian_power", "10")
    config_sop.set("ShapeGradient", "p_laplacian_stabilization", "0.0")

    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J, u, p, geometry.boundaries, config=config_sop
    )
    sop.solve(algorithm="gd", rtol=1e-1, max_iter=6)

    assert sop.solver.relative_norm <= 1e-1


def test_p_laplacian_iterative(rng, config_sop, e, bcs, J, u, p, geometry):
    config_sop.set("ShapeGradient", "mu_def", "1.0")
    config_sop.set("ShapeGradient", "mu_fix", "1.0")
    config_sop.set("ShapeGradient", "damping_factor", "1.0")
    config_sop.set("ShapeGradient", "use_p_laplacian", "True")
    config_sop.set("ShapeGradient", "p_laplacian_power", "10")
    config_sop.set("ShapeGradient", "p_laplacian_stabilization", "0.0")

    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J, u, p, geometry.boundaries, config=config_sop
    )
    assert sop.gradient_test(rng=rng) > 1.9
    assert sop.gradient_test(rng=rng) > 1.9
    assert sop.gradient_test(rng=rng) > 1.9
