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

"""For a documentation of this demo see demo_04_multiple_variables.md

"""

from fenics import *

import cashocs



config = cashocs.create_config('config.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
z = Function(V)
p = Function(V)
q = Function(V)
u = Function(V)
v = Function(V)

states = [y, z]
adjoints = [p, q]
controls = [u, v]

e_y = inner(grad(y), grad(p)) * dx - u * p * dx
e_z = inner(grad(z), grad(q)) * dx - (y + v) * q * dx

e = [e_y, e_z]

bcs1 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs2 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

bcs_list = [bcs1, bcs2]

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
alpha = 1e-6
beta = 1e-4
J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5) * (z - z_d) * (z - z_d) * dx \
	+ Constant(0.5*alpha) * u * u * dx + Constant(0.5*beta) * v * v * dx

ocp = cashocs.OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)
ocp.solve()



### Post Processing

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))

plt.subplot(2, 3, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('Control variable u')

plt.subplot(2, 3, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('State variable y')

plt.subplot(2, 3, 3)
fig = plot(y_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('Desired state y_d')

plt.subplot(2, 3, 4)
fig = plot(v)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('Control variable v')

plt.subplot(2, 3, 5)
fig = plot(z)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('State variable z')

plt.subplot(2, 3, 6)
fig = plot(z_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('Desired state z_d')

plt.tight_layout()
# plt.savefig('./img_multiple_variables.png', dpi=150, bbox_inches='tight')
