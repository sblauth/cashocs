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

"""Tests for the nonlinear_solvers module."""

from fenics import *
import numpy as np

import cashocs
import cashocs._utils
import cashocs._utils.linalg


def test_newton_solver():
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    u_fen = Function(V)
    v = TestFunction(V)

    F = (
        inner(grad(u), grad(v)) * dx
        + Constant(1e2) * pow(u, 3) * v * dx
        - Constant(1) * v * dx
    )
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    solve(F == 0, u, bcs)
    u_fen.vector().vec().aypx(0.0, u.vector().vec())
    u_fen.vector().apply("")
    u.vector().vec().set(0.0)
    u.vector().apply("")
    cashocs.newton_solve(
        F,
        u,
        bcs,
        rtol=1e-9,
        atol=1e-10,
        max_iter=50,
        convergence_type="combined",
        norm_type="l2",
        damped=False,
        verbose=True,
    )

    assert np.allclose(u.vector()[:], u_fen.vector()[:])


def test_snes_solver():
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    u_fen = Function(V)
    v = TestFunction(V)

    F = (
        inner(grad(u), grad(v)) * dx
        + Constant(1e2) * pow(u, 3) * v * dx
        - Constant(1) * v * dx
    )
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    solve(F == 0, u, bcs)
    u_fen.vector().vec().aypx(0.0, u.vector().vec())
    u_fen.vector().apply("")
    u.vector().vec().set(0.0)
    u.vector().apply("")

    petsc_options = {
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_atol": 1e-10,
        "snes_rtol": 1e-9,
    }
    petsc_options.update(cashocs._utils.linalg.direct_ksp_options)

    cashocs.snes_solve(F, u, bcs, petsc_options=petsc_options)

    assert np.allclose(u.vector()[:], u_fen.vector()[:])


def test_ts_pseudo_solver():
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    u_fen = Function(V)
    v = TestFunction(V)

    F = (
        inner(grad(u), grad(v)) * dx
        + Constant(1e2) * pow(u, 3) * v * dx
        - Constant(1) * v * dx
    )
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    solve(F == 0, u, bcs)
    u_fen.vector().vec().aypx(0.0, u.vector().vec())
    u_fen.vector().apply("")
    u.vector().vec().set(0.0)
    u.vector().apply("")

    petsc_options = {
        "ts_type": "beuler",
        "ts_dt": 1e-1,
        "ts_max_steps": 100,
        "snes_type": "ksponly",
    }
    petsc_options.update(cashocs._utils.linalg.direct_ksp_options)

    cashocs.ts_pseudo_solve(
        F, u, bcs, petsc_options=petsc_options, rtol=1e-9, atol=1e-10
    )

    assert np.allclose(u.vector()[:], u_fen.vector()[:])


def test_newton_linearization(config_sop):
    config_sop.set("StateSystem", "is_linear", "False")
    config_sop.set("StateSystem", "newton_verbose", "True")

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(16)
    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

    up = Function(V)
    u, p = split(up)
    vq = Function(V)
    v, q = split(vq)

    F = (
        Constant(1e-2) * inner(grad(u), grad(v)) * dx
        + dot(grad(u) * u, v) * dx
        - p * div(v) * dx
        - q * div(u) * dx
    )
    bcs = cashocs.create_dirichlet_bcs(V.sub(0), Constant((1.0, 0.0)), boundaries, 1)
    bcs += cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0.0, 0.0)), boundaries, [3, 4]
    )

    u_, p_ = TrialFunctions(V)
    v_, q_ = TestFunctions(V)
    dF = (
        Constant(1e-2) * inner(grad(u_), grad(v_)) * dx
        + dot(grad(u_) * u, v_) * dx
        - p_ * div(v_) * dx
        - q_ * div(u_) * dx
    )

    J = cashocs.IntegralFunctional(Constant(0.0) * dx)
    sop = cashocs.ShapeOptimizationProblem(
        F,
        bcs,
        J,
        up,
        vq,
        boundaries,
        config=config_sop,
        newton_linearizations=dF,
    )
    sop.compute_state_variables()
