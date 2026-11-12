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

"""Tests for the nonlinear_solvers module."""

from fenics import *
import numpy as np
import pytest

import cashocs
import cashocs._utils
import cashocs._utils.linalg
from cashocs.nonlinear_solvers.ts import TSPseudoSolver


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


def test_ts_pseudo_solver_residual_cache():
    mesh, _, _, dx, _, _ = cashocs.regular_mesh(2)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)
    nonlinear_form = u * v * dx
    shift = Constant(1.0) * v * dx

    solver = TSPseudoSolver(nonlinear_form, u, [], shift=shift)

    class CountingAssembler:
        def __init__(self, assembler):
            self.assembler = assembler
            self.calls = 0

        def assemble(self, *args, **kwargs):
            self.calls += 1
            return self.assembler.assemble(*args, **kwargs)

    assembler = CountingAssembler(solver.assembler)
    assembler_shift = CountingAssembler(solver.assembler_shift)
    solver.assembler = assembler
    solver.assembler_shift = assembler_shift

    f = u.vector().vec().duplicate()
    try:
        residual_norm = solver.compute_nonlinear_residual(u.vector().vec())
        cached_residual = solver.residual_convergence.vec().getArray().copy()
        assert assembler.calls == 1
        assert assembler_shift.calls == 1

        assert residual_norm == pytest.approx(solver.residual_convergence.norm("l2"))
        solver.assemble_residual(None, 0.0, u.vector().vec(), f)
        assert np.allclose(f.getArray(), -cached_residual)
        assert assembler.calls == 1
        assert assembler_shift.calls == 1

        f.set(123.0)
        assert solver.compute_nonlinear_residual(u.vector().vec()) == pytest.approx(
            residual_norm
        )
        assert np.allclose(
            solver.residual_convergence.vec().getArray(), cached_residual
        )
        assert assembler.calls == 1
        assert assembler_shift.calls == 1

        u.vector().vec().set(0.25)
        u.vector().apply("")
        solver.compute_nonlinear_residual(u.vector().vec())
        assert assembler.calls == 2
        assert assembler_shift.calls == 2
    finally:
        f.destroy()
        solver._destroy_petsc_objects()
