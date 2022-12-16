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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/optimal_control/doc_dirichlet_control.html.

"""

from fenics import *
import numpy as np

import cashocs

# load mesh and config
config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(32)
n = FacetNormal(mesh)
h = MaxCellEdgeLength(mesh)
V = FunctionSpace(mesh, "CG", 1)

# define state, adjoint, and control variables
y = Function(V)
p = Function(V)
u = Function(V)

# define the PDE constraint
# Note: Due to the boundary conditions, the second term vanishes.
# Note: The second term is used to get the correct gradient for the Dirichlet boundary
# condition, which is dot(grad(p), n), where p is the adjoint state.
e = dot(grad(y), grad(p)) * dx - dot(grad(p), n) * (y - u) * ds - Constant(1) * p * dx
bcs = cashocs.create_dirichlet_bcs(V, u, boundaries, [1, 2, 3, 4])

# define the cost functional
y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
alpha = 1e-4
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * ds
)

# define a custom scalar product for L^2(\Gamma)
scalar_product = TrialFunction(V) * TestFunction(V) * ds

# define and solve the optimization problem
ocp = cashocs.OptimalControlProblem(
    e, bcs, J, y, u, p, config=config, riesz_scalar_products=scalar_product
)
ocp.solve()


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
# plt.savefig(
#     "./img_dirichlet_control_via_lagrange_multiplier.png", dpi=150, bbox_inches="tight"
# )
