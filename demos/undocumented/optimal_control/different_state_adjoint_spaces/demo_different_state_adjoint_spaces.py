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

"""This demo shows how to use different discretizations for state and
adjoint system.

"""

from fenics import *

import cashocs

set_log_level(LogLevel.CRITICAL)
config = cashocs.load_config("config.ini")
parameters["ghost_mode"] = "shared_vertex"

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)

# Create different spaces for state and adjoint variables
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 1)

y = Function(V)
p = Function(W)
u = Function(V)

# Set up a discontinuous Galerkin method (SIPG) (needed for the adjoint system,
# reduces to the classical Galerkin method for CG elements)
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2
flux = 1 / 2 * (inner(grad(u)("+") + grad(u)("-"), n("+")))
alpha = 1e3
gamma = 1e3
e = (
    dot(grad(p), grad(y)) * dx
    - dot(avg(grad(p)), jump(y, n)) * dS
    - dot(jump(p, n), avg(grad(y))) * dS
    + Constant(alpha) / h_avg * dot(jump(p, n), jump(y, n)) * dS
    - dot(grad(p), y * n) * ds
    - dot(p * n, grad(y)) * ds
    + (Constant(gamma) / h) * p * y * ds
    - u * p * dx
)

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

lambd = 1e-6
y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)

J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * lambd) * u * u * dx
)

optimization_problem = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config=config)
optimization_problem.solve()
