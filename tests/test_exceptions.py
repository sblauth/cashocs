# Copyright (C) 2020 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Tests if CASHOCS exceptions are being raised properly.

"""

import os

import numpy as np
import pytest
from fenics import *

import cashocs
from cashocs._exceptions import (
    CashocsException,
    ConfigError,
    InputError,
    NotConvergedError,
    PETScKSPError,
)

rng = np.random.RandomState(300696)
dir_path = os.path.dirname(os.path.realpath(__file__))
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(6)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx

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
    with pytest.raises(NotConvergedError) as e_info:
        u.vector()[:] = 0.0
        ocp._erase_pde_memory()
        ocp.solve("gd", 1e-10, 0.0, 1)
    assert "failed to converge" in str(e_info.value)

    with pytest.raises(CashocsException):
        ocp.solve("gd", 1e-10, 0.0, 0)


def test_input_error():
    with pytest.raises(InputError) as e_info:
        cashocs.regular_mesh(-1)
    assert "Not a valid input for object" in str(e_info.value)

    with pytest.raises(CashocsException):
        cashocs.regular_mesh(0)


def test_petsc_error():
    with pytest.raises(PETScKSPError) as e_info:
        u.vector()[:] = rng.rand(V.dim())
        ocp_ksp._erase_pde_memory()
        ocp_ksp.compute_state_variables()
    assert "PETSc linear solver did not converge." in str(e_info.value)

    with pytest.raises(CashocsException):
        u.vector()[:] = rng.rand(V.dim())
        ocp_ksp._erase_pde_memory()
        ocp_ksp.compute_state_variables()


def test_config_error():
    with pytest.raises(ConfigError) as e_info:
        config.set("AlgoCG", "cg_method", "nonexistent")
        config.set("OptimizationRoutine", "algorithm", "cg")
        config.set("OptimizationRoutine", "maximum_iterations", "2")
        u.vector()[:] = rng.rand(V.dim())
        ocp_conf = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
        ocp_conf.solve(max_iter=10)
    assert "You have an error in your config file." in str(e_info.value)

    with pytest.raises(CashocsException):
        ocp_conf._erase_pde_memory()
        ocp_conf.solve(max_iter=10)
