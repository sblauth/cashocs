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

import pathlib

from fenics import *
import numpy as np
import pytest

import cashocs
from cashocs._exceptions import CashocsException
from cashocs._exceptions import InputError
from cashocs._exceptions import NotConvergedError
from cashocs._exceptions import PETScKSPError

rng = np.random.RandomState(300696)
dir_path = str(pathlib.Path(__file__).parent)
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(6)
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

ksp_options = [
    ["ksp_type", "cg"],
    ["pc_type", "hypre"],
    ["pc_hypre_type", "boomeramg"],
    ["ksp_rtol", 0.0],
    ["ksp_atol", 0.0],
    ["ksp_max_it", 1],
    ["ksp_monitor_true_residual"],
]

ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
ocp_ksp = cashocs.OptimalControlProblem(
    F, bcs, J, y, u, p, config, ksp_options=ksp_options
)


def test_not_converged_error():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    with pytest.raises(NotConvergedError) as e_info:
        u.vector().vec().set(0.0)
        u.vector().apply("")
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)
        ocp.solve(algorithm="gd", rtol=1e-10, atol=0.0, max_iter=1)
    MPI.barrier(MPI.comm_world)
    assert "failed to converge" in str(e_info.value)

    with pytest.raises(CashocsException):
        config = cashocs.load_config(dir_path + "/config_ocp.ini")
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)
        ocp.solve(algorithm="gd", rtol=1e-10, atol=0.0, max_iter=0)
    MPI.barrier(MPI.comm_world)


def test_input_error():
    with pytest.raises(InputError) as e_info:
        bcs = [None]
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    MPI.barrier(MPI.comm_world)
    assert "Not a valid input for object" in str(e_info.value)


def test_petsc_error():
    with pytest.raises(PETScKSPError) as e_info:
        u.vector().set_local(rng.rand(u.vector().local_size()))
        u.vector().apply("")
        ocp_ksp._erase_pde_memory()
        ocp_ksp.compute_state_variables()
    MPI.barrier(MPI.comm_world)
    assert "PETSc linear solver did not converge." in str(e_info.value)

    with pytest.raises(CashocsException):
        u.vector().set_local(rng.rand(u.vector().local_size()))
        u.vector().apply("")
        ocp_ksp._erase_pde_memory()
        ocp_ksp.compute_state_variables()
    MPI.barrier(MPI.comm_world)
