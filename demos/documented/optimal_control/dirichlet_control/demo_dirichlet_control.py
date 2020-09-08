# Copyright (C) 2020 Sebastian Blauth
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

"""For the documentation of this demo, see doc_dirichlet_control.md

"""

import numpy as np
from fenics import *

import cashocs



config = cashocs.create_config('config.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
n = FacetNormal(mesh)
h = MaxCellEdgeLength(mesh)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
p = Function(V)
u = Function(V)

bcs = []

eta = Constant(1e4)
e = inner(grad(y), grad(p))*dx - inner(grad(y), n)*p*ds - inner(grad(p), n)*(y - u)*ds + eta/h*(y - u)*p*ds - Constant(1)*p*dx

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
alpha = 1e-4
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*ds

scalar_product = TrialFunction(V)*TestFunction(V)*ds

ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config, riesz_scalar_products=scalar_product)
ocp.solve()


bcs = cashocs.create_bcs_list(V, 1, boundaries, [1,2,3,4])
bdry_idx = Function(V)
[bc.apply(bdry_idx.vector()) for bc in bcs]
mask = np.where(bdry_idx.vector()[:] == 1)[0]

y_bdry = Function(V)
u_bdry = Function(V)
y_bdry.vector()[mask] = y.vector()[mask]
u_bdry.vector()[mask] = u.vector()[mask]

error_inf = np.max(np.abs(y_bdry.vector()[:] - u_bdry.vector()[:])) / np.max(np.abs(u_bdry.vector()[:])) * 100
error_l2 = np.sqrt(assemble((y - u)*(y - u)*ds)) / np.sqrt(assemble(u*u*ds)) * 100

print('Error regarding the (weak) imposition of the boundary values')
print('Error L^\infty: ' + format(error_inf, '.3e') + ' %')
print('Error L^2: ' + format(error_l2, '.3e') + ' %')



### Post Processing
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15,5))
#
# plt.subplot(1, 3, 1)
# fig = plot(u)
# plt.colorbar(fig, fraction=0.046, pad=0.04)
# plt.title('Control variable u')
#
# plt.subplot(1,3,2)
# fig = plot(y)
# plt.colorbar(fig, fraction=0.046, pad=0.04)
# plt.title('State variable y')
#
# plt.subplot(1,3,3)
# fig = plot(y_d, mesh=mesh)
# plt.colorbar(fig, fraction=0.046, pad=0.04)
# plt.title('Desired state y_d')
#
# plt.tight_layout()
# plt.savefig('./img_dirichlet_control.png', dpi=150, bbox_inches='tight')
