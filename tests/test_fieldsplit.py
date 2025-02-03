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

"""Tests for the linear solvers."""

from fenics import *
import numpy as np

import cashocs


def test_fieldsplit_pc():
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(8)
    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement(v_elem, p_elem))

    up = Function(V)
    u, p = split(up)
    v, q = TestFunctions(V)

    F = inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx
    bcs = cashocs.create_dirichlet_bcs(V.sub(0), Constant((1.0, 0.0)), boundaries, 1)
    bcs += cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0.0, 0.0)), boundaries, [3, 4]
    )

    ksp_options = {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-5,
        "ksp_max_it": 1,
        "ksp_monitor_true_residual": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_precondition": "selfp",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_1_ksp_rtol": 1e-6,
        "fieldsplit_1_ksp_max_it": 25,
        "fieldsplit_1_pc_type": "hypre",
        "fieldsplit_1_ksp_converged_reason": None,
    }

    cashocs.linear_solve(F, up, bcs, ksp_options=ksp_options)
    assert True


def test_fieldsplit_pc_section():
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(8)
    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement(p_elem, v_elem))

    pu = Function(V)
    p, u = split(pu)
    q, v = TestFunctions(V)

    F = inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx
    bcs = cashocs.create_dirichlet_bcs(V.sub(1), Constant((1.0, 0.0)), boundaries, 1)
    bcs += cashocs.create_dirichlet_bcs(
        V.sub(1), Constant((0.0, 0.0)), boundaries, [3, 4]
    )

    ksp_options = {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-5,
        "ksp_max_it": 1,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_precondition": "selfp",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0",
        "fieldsplit_0_ksp_type": "gmres",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_0_ksp_rtol": 1e-6,
        "fieldsplit_0_ksp_max_it": 25,
        "fieldsplit_0_ksp_converged_reason": None,
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "lu",
    }

    cashocs.linear_solve(F, pu, bcs, ksp_options=ksp_options)
    assert True


def test_fieldsplit_snes_nested():
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(8)
    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement(v_elem, p_elem, p_elem))

    upT = Function(V)
    u, p, T = split(upT)
    v, q, S = TestFunctions(V)

    mu = 1.0 / (T + 1)

    F = (
        mu * inner(grad(u), grad(v)) * dx
        + dot(grad(u) * u, v) * dx
        - p * div(v) * dx
        - q * div(u) * dx
        + dot(grad(T), grad(S)) * dx
        + dot(u, grad(T)) * S * dx
        - Constant(1.0) * S * dx
    )
    bcs = cashocs.create_dirichlet_bcs(V.sub(0), Constant((1.0, 0.0)), boundaries, 1)
    bcs += cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0.0, 0.0)), boundaries, [3, 4]
    )
    bcs += cashocs.create_dirichlet_bcs(V.sub(2), Constant(1.0), boundaries, [1, 3, 4])

    petsc_options = {
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_max_it": 7,
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-1,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
        "pc_fieldsplit_0_fields": "0,1",
        "pc_fieldsplit_1_fields": "2",
        "fieldsplit_0_ksp_type": "fgmres",
        "fieldsplit_0_ksp_rtol": 1e-1,
        "fieldsplit_0_pc_type": "fieldsplit",
        "fieldsplit_0_pc_fieldsplit_type": "schur",
        "fieldsplit_0_pc_fieldsplit_schur_precondition": "selfp",
        "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_0_fieldsplit_1_ksp_rtol": 1e-1,
        "fieldsplit_0_fieldsplit_1_pc_type": "hypre",
        "fieldsplit_0_fieldsplit_1_ksp_converged_reason": None,
        "fieldsplit_2_ksp_type": "gmres",
        "fieldsplit_2_ksp_rtol": 1e-1,
        "fieldsplit_2_pc_type": "hypre",
    }

    cashocs.snes_solve(F, upT, bcs, petsc_options=petsc_options, max_iter=8)
    assert True
