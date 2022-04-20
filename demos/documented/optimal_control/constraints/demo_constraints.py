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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/optimal_control/doc_constraints.html.

"""

from fenics import *
import numpy as np

import cashocs

cashocs.set_log_level(cashocs.LogLevel.INFO)
config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(32)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

e = inner(grad(y), grad(p)) * dx - u * p * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
alpha = 1e-4
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

bottom_left = Expression("(x[0] <= 0.5) && (x[1] <= 0.5) ? 1.0 : 0.0", degree=0)
bottom_right = Expression("(x[0] >= 0.5) && (x[1] <= 0.5) ? 1.0 : 0.0", degree=0)
top_left = Expression("(x[0] <= 0.5) && (x[1] >= 0.5) ? 1.0 : 0.0", degree=0)
top_right = Expression("(x[0] >= 0.5) && (x[1] >= 0.5) ? 1.0 : 0.0", degree=0)

pointwise_equality_constraint = cashocs.EqualityConstraint(
    bottom_left * y, 0.0, measure=dx
)
integral_equality_constraint = cashocs.EqualityConstraint(
    bottom_right * pow(y, 2) * dx, 0.01
)
integral_inequality_constraint = cashocs.InequalityConstraint(
    top_left * y * dx, lower_bound=-0.025
)
pointwise_inequality_constraint = cashocs.InequalityConstraint(
    top_right * y, upper_bound=0.25, measure=dx
)

constraints = [
    pointwise_equality_constraint,
    integral_equality_constraint,
    integral_inequality_constraint,
    pointwise_inequality_constraint,
]

problem = cashocs.ConstrainedOptimalControlProblem(
    e, bcs, J, y, u, p, constraints, config
)
problem.solve(method="AL")


### Post Processing

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable u")

plt.subplot(1, 3, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable y")

plt.subplot(1, 3, 3)
fig = plot(y_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Desired state y_d")

plt.tight_layout()
# plt.savefig("./img_constraints.png", dpi=150, bbox_inches="tight")
