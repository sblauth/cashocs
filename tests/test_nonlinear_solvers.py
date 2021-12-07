# Copyright (C) 2020-2021 Sebastian Blauth
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

"""Tests for the nonlinear_solvers module.

"""

import numpy as np
from fenics import *

import cashocs



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
    u_fen.vector()[:] = u.vector()[:]
    u.vector()[:] = 0.0
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
