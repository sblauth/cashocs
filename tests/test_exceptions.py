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

"""Tests if cashocs exceptions are being raised properly.

"""

from collections import namedtuple

from fenics import *
import pytest

import cashocs
from cashocs._exceptions import CashocsException
from cashocs._exceptions import InputError
from cashocs._exceptions import NotConvergedError
from cashocs._exceptions import PETScKSPError


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
def u(CG1):
    return Function(CG1)


@pytest.fixture
def p(CG1):
    return Function(CG1)


@pytest.fixture
def F(y, u, p, geometry):
    return dot(grad(y), grad(p)) * geometry.dx - u * p * geometry.dx


@pytest.fixture
def bcs(CG1, geometry):
    return cashocs.create_dirichlet_bcs(
        CG1, Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )


@pytest.fixture
def J(y, y_d, u, geometry):
    alpha = 1e-6
    return cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
    )


@pytest.fixture
def ksp_options():
    return [
        ["ksp_type", "cg"],
        ["pc_type", "hypre"],
        ["pc_hypre_type", "boomeramg"],
        ["ksp_rtol", 0.0],
        ["ksp_atol", 0.0],
        ["ksp_max_it", 1],
        ["ksp_monitor_true_residual"],
    ]


@pytest.fixture
def ocp(F, bcs, J, y, u, p, config_ocp):
    return cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)


@pytest.fixture
def ocp_ksp(F, bcs, J, y, u, p, config_ocp, ksp_options):
    return cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config=config_ocp, ksp_options=ksp_options
    )


def test_not_converged_error(ocp):
    with pytest.raises(NotConvergedError) as e_info:
        ocp.solve(algorithm="gd", rtol=1e-10, atol=0.0, max_iter=1)
    MPI.barrier(MPI.comm_world)
    assert "failed to converge" in str(e_info.value)

    with pytest.raises(CashocsException):
        ocp.solve(algorithm="gd", rtol=1e-10, atol=0.0, max_iter=0)
    MPI.barrier(MPI.comm_world)


def test_input_error(F, J, y, u, p, config_ocp):
    with pytest.raises(InputError) as e_info:
        bcs = [None]
        cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    MPI.barrier(MPI.comm_world)
    assert "Not a valid input for object" in str(e_info.value)


def test_petsc_error(ocp_ksp, u, rng):
    with pytest.raises(PETScKSPError) as e_info:
        u.vector().set_local(rng.rand(u.vector().local_size()))
        u.vector().apply("")
        ocp_ksp.compute_state_variables()
    MPI.barrier(MPI.comm_world)
    assert "PETSc linear solver did not converge." in str(e_info.value)

    with pytest.raises(CashocsException):
        u.vector().set_local(rng.rand(u.vector().local_size()))
        u.vector().apply("")
        ocp_ksp.compute_state_variables()
    MPI.barrier(MPI.comm_world)
