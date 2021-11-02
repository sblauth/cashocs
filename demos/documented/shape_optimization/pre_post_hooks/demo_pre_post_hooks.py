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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_pre_post_hooks.html.

"""

from fenics import *

import cashocs

config = cashocs.load_config("./config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh("./mesh/mesh.xdmf")

v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

Re = 1e2
e = (
    inner(grad(u), grad(v)) * dx
    + Constant(Re) * dot(grad(u) * u, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
)

u_in = Expression(("-1.0/4.0*(x[1] - 2.0)*(x[1] + 2.0)", "0.0"), degree=2)
bc_in = DirichletBC(V.sub(0), u_in, boundaries, 1)
bc_no_slip = cashocs.create_dirichlet_bcs(
    V.sub(0), Constant((0, 0)), boundaries, [2, 4]
)
bcs = [bc_in] + bc_no_slip

J = Constant(1) * inner(grad(u), grad(u)) * dx

sop = cashocs.ShapeOptimizationProblem(e, bcs, J, up, vq, boundaries, config)
sop.solve()


### Post Processing

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 3))

ax_mesh = plt.subplot(1, 3, 1)
fig_mesh = plot(mesh)
plt.title("Discretization of the optimized geometry")

ax_u = plt.subplot(1, 3, 2)
ax_u.set_xlim(ax_mesh.get_xlim())
ax_u.set_ylim(ax_mesh.get_ylim())
fig_u = plot(u)
plt.colorbar(fig_u, fraction=0.046, pad=0.04)
plt.title("State variable u")


ax_p = plt.subplot(1, 3, 3)
ax_p.set_xlim(ax_mesh.get_xlim())
ax_p.set_ylim(ax_mesh.get_ylim())
fig_p = plot(p)
plt.colorbar(fig_p, fraction=0.046, pad=0.04)
plt.title("State variable p")

plt.tight_layout()
# plt.savefig('./img_pre_post_hooks.png', dpi=150, bbox_inches='tight')
