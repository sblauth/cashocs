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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/optimal_control/doc_monolithic_problems.html.

"""

from fenics import *

import cashocs



config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)

elem_1 = FiniteElement("CG", mesh.ufl_cell(), 1)
elem_2 = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([elem_1, elem_2]))

U = FunctionSpace(mesh, "CG", 1)

state = Function(V)
adjoint = Function(V)
y, z = split(state)
p, q = split(adjoint)

u = Function(U)
v = Function(U)
controls = [u, v]

e_y = inner(grad(y), grad(p)) * dx + z * p * dx - u * p * dx
bcs_y = cashocs.create_dirichlet_bcs(V.sub(0), Constant(0), boundaries, [1, 2, 3, 4])

e_z = inner(grad(z), grad(q)) * dx + y * q * dx - v * q * dx
bcs_z = cashocs.create_dirichlet_bcs(V.sub(1), Constant(0), boundaries, [1, 2, 3, 4])

e = e_y + e_z
bcs = bcs_y + bcs_z

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
z_d = Expression("sin(4*pi*x[0])*sin(4*pi*x[1])", degree=1)
alpha = 1e-6
beta = 1e-6
J = (
    Constant(0.5) * (y - y_d) * (y - y_d) * dx
    + Constant(0.5) * (z - z_d) * (z - z_d) * dx
    + Constant(0.5 * alpha) * u * u * dx
    + Constant(0.5 * beta) * v * v * dx
)

optimization_problem = cashocs.OptimalControlProblem(
    e, bcs, J, state, controls, adjoint, config
)
optimization_problem.solve()


### Post Processing

y, z = state.split(True)
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable u")

plt.subplot(2, 3, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable y")

plt.subplot(2, 3, 3)
fig = plot(y_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Desired state y_d")

plt.subplot(2, 3, 4)
fig = plot(v)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable v")

plt.subplot(2, 3, 5)
fig = plot(z)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable z")

plt.subplot(2, 3, 6)
fig = plot(z_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Desired state z_d")

plt.tight_layout()
# plt.savefig('./img_monolithic_problems.png', dpi=150, bbox_inches='tight')
