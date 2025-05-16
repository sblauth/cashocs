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

import pathlib

from fenics import *
from mpi4py import MPI
import numpy as np

import cashocs

rng = np.random.RandomState(300696)
dir_path = str(pathlib.Path(__file__).parent)

mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
    f"{dir_path}/mesh/unit_circle/mesh.xdmf"
)
initial_coordinates = mesh.coordinates().copy()

V = FunctionSpace(mesh, "CG", 1)

bcs = DirichletBC(V, Constant(0), boundaries, 1)

x = SpatialCoordinate(mesh)
f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

u = Function(V)
p = Function(V)

e = inner(grad(u), grad(p)) * dx - f * p * dx

J = cashocs.IntegralFunctional(u * dx)


def test_int_eq_constraints():
    cfg = cashocs.load_config(dir_path + "/config_sop.ini")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    constraint = cashocs.EqualityConstraint(1 * dx, 1.0)
    problem = cashocs.ConstrainedShapeOptimizationProblem(
        e, bcs, J, u, p, boundaries, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3


def test_int_ineq_constraints():
    cfg = cashocs.load_config(dir_path + "/config_sop.ini")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    constraint = cashocs.InequalityConstraint(1 * dx, upper_bound=1.5, lower_bound=0.0)
    problem = cashocs.ConstrainedShapeOptimizationProblem(
        e, bcs, J, u, p, boundaries, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2, mu_0=1e-0)
    assert constraint.constraint_violation() <= 1e-3

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3
