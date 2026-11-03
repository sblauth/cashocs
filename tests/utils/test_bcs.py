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

from collections import namedtuple
import pathlib

from fenics import *
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs._exceptions import InputError
import cashocs._utils
import cashocs._utils.linalg


@pytest.fixture
def geometry():
    Geometry = namedtuple("Geometry", "mesh boundaries dx ds")
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
    geom = Geometry(mesh, boundaries, dx, ds)

    return geom


@pytest.fixture
def V(geometry):
    V = FunctionSpace(geometry.mesh, "CG", 1)

    return V


def evaluate_function(u, x, geometry):
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = geometry.mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None

    # Gather the results on process 0
    comm = geometry.mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array(
            [y for y in computed_u if y is not None], dtype=np.double
        )
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # Broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u


def test_create_bcs(V, geometry):
    rng = np.random.RandomState(300696)

    trial = TrialFunction(V)
    test = TestFunction(V)
    a = inner(grad(trial), grad(test)) * dx
    L = Constant(1) * test * dx

    bc_val = rng.rand()
    bc1 = DirichletBC(V, Constant(bc_val), geometry.boundaries, 1)
    bc2 = DirichletBC(V, Constant(bc_val), geometry.boundaries, 2)
    bc3 = DirichletBC(V, Constant(bc_val), geometry.boundaries, 3)
    bc4 = DirichletBC(V, Constant(bc_val), geometry.boundaries, 4)
    bcs_ex = [bc1, bc2, bc3, bc4]
    bcs = cashocs.create_dirichlet_bcs(
        V, Constant(bc_val), geometry.boundaries, [1, 2, 3, 4]
    )

    u_ex = Function(V)
    u = Function(V)

    solve(a == L, u_ex, bcs_ex)
    solve(a == L, u, bcs)

    assert np.allclose(u_ex.vector()[:], u.vector()[:])
    assert abs(evaluate_function(u, (0, 0), geometry) - bc_val) < 1e-14


def test_create_named_bcs(copy_test_asset, rng):
    mesh_file = copy_test_asset("data/meshes/unit_named/named_mesh.msh")
    xdmf_file = mesh_file.with_suffix(".xdmf")
    cashocs.convert(str(mesh_file), str(xdmf_file))

    mesh_, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(xdmf_file))
    V_ = FunctionSpace(mesh_, "CG", 1)

    bcs_str = cashocs.create_dirichlet_bcs(
        V_, Constant(0.0), boundaries, ["inlet", "wall", "outlet"]
    )
    bcs_int = cashocs.create_dirichlet_bcs(V_, Constant(0.0), boundaries, [1, 2, 3])
    bcs_mixed = cashocs.create_dirichlet_bcs(
        V_, Constant(0.0), boundaries, ["inlet", 2, "outlet"]
    )

    fun1 = Function(V_)
    fun1.vector().set_local(rng.rand(fun1.vector().local_size()))
    fun1.vector().apply("")

    fun2 = Function(V_)
    fun2.vector().vec().aypx(0.0, fun1.vector().vec())
    fun2.vector().apply("")

    fun3 = Function(V_)
    fun3.vector().vec().aypx(0.0, fun1.vector().vec())
    fun3.vector().apply("")

    [bc.apply(fun1.vector()) for bc in bcs_int]
    [bc.apply(fun2.vector()) for bc in bcs_str]
    [bc.apply(fun3.vector()) for bc in bcs_mixed]

    for i in range(len(bcs_str)):
        assert np.max(np.abs(fun1.vector()[:] - fun2.vector()[:])) <= 1e-14
        assert np.max(np.abs(fun1.vector()[:] - fun3.vector()[:])) <= 1e-14
        assert np.max(np.abs(fun2.vector()[:] - fun3.vector()[:])) <= 1e-14

    MPI.barrier(MPI.comm_world)
    assert xdmf_file.is_file()
    assert xdmf_file.with_suffix(".h5").is_file()
    assert xdmf_file.with_name("named_mesh_boundaries.xdmf").is_file()
    assert xdmf_file.with_name("named_mesh_boundaries.h5").is_file()
    assert xdmf_file.with_name("named_mesh_physical_groups.json").is_file()
    MPI.barrier(MPI.comm_world)
