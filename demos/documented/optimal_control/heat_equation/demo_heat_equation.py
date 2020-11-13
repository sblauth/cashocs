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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/optimal_control/doc_heat_equation.html.

"""

import numpy as np
from fenics import *

import cashocs



config = cashocs.load_config('config.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
V = FunctionSpace(mesh, 'CG', 1)

dt = 1 / 10
t_start = dt
t_end = 1.0
t_array = np.linspace(t_start, t_end, int(1/dt))

states = [Function(V) for i in range(len(t_array))]
controls = [Function(V) for i in range(len(t_array))]
adjoints = [Function(V) for i in range(len(t_array))]

bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
bcs_list = [bcs for i in range(len(t_array))]

y_d = []
e = []
J_list = []

alpha = 1e-5
y_d_expr = Expression('exp(-20*(pow(x[0] - 0.5 - 0.25*cos(2*pi*t), 2) + pow(x[1] - 0.5 - 0.25*sin(2*pi*t), 2)))', degree=1, t=0.0)



for k in range(len(t_array)):
	t = t_array[k]
	y_d_expr.t = t

	y = states[k]
	if k == 0:
		y_prev = Function(V)
	else:
		y_prev = states[k - 1]
	p = adjoints[k]
	u = controls[k]

	state_eq = Constant(1/dt)*(y - y_prev)*p*dx + inner(grad(y), grad(p))*dx - u*p*dx

	e.append(state_eq)
	y_d.append(interpolate(y_d_expr, V))

	J_list.append(Constant(0.5*dt) * (y - y_d[k]) * (y - y_d[k]) * dx + Constant(0.5 * dt * alpha) * u * u * dx)


J = cashocs.utils.summation(J_list)

ocp = cashocs.OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)
ocp.solve()



### Post processing

u_file = File('./visualization/u.pvd')
y_file = File('./visualization/y.pvd')
temp_u = Function(V)
temp_y = Function(V)

for k in range(len(t_array)):
	t = t_array[k]

	temp_u.vector()[:] = controls[k].vector()[:]
	u_file << temp_u, t

	temp_y.vector()[:] = states[k].vector()[:]
	y_file << temp_y, t
