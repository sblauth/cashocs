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

"""Tests for the module PDE problems.

"""

import pathlib

from fenics import *
import numpy as np

import cashocs

rng = np.random.RandomState(300696)
dir_path = str(pathlib.Path(__file__).parent)
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

e = inner(grad(y), grad(p)) * dx - u * p * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2])

y_d = Function(V)
alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)


def test_state_adjoint_problems():
    trial = TrialFunction(V)
    test = TestFunction(V)
    state = Function(V)
    adjoint = Function(V)

    a = inner(grad(trial), grad(test)) * dx
    L_state = u * test * dx
    L_adjoint = -(state - y_d) * test * dx

    y_d.vector().set_local(rng.rand(y_d.vector().local_size()))
    y_d.vector().apply("")
    u.vector().set_local(rng.rand(u.vector().local_size()))
    u.vector().apply("")

    ocp.compute_state_variables()
    ocp.compute_adjoint_variables()

    solve(a == L_state, state, bcs)
    solve(a == L_adjoint, adjoint, bcs)

    assert np.allclose(state.vector()[:], y.vector()[:])
    assert np.allclose(adjoint.vector()[:], p.vector()[:])


def test_control_gradient():
    trial = TrialFunction(V)
    test = TestFunction(V)
    gradient = Function(V)

    a = trial * test * dx
    L = Constant(alpha) * u * test * dx - p * test * dx

    ocp._erase_pde_memory()
    y_d.vector().set_local(rng.rand(y_d.vector().local_size()))
    y_d.vector().apply("")
    u.vector().set_local(rng.rand(u.vector().local_size()))
    u.vector().apply("")

    c_gradient = ocp.compute_gradient()[0]
    solve(a == L, gradient)

    assert np.allclose(c_gradient.vector()[:], gradient.vector()[:])
