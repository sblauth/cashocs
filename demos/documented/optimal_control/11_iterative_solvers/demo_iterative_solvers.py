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

"""This demo shows the interface for specifying the linear solvers. The documentation
can be found in doc_iterative_solvers.md

"""

from fenics import *
import cashocs



config = cashocs.create_config('config.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
p = Function(V)
u = Function(V)

e = inner(grad(y), grad(p))*dx - u*p*dx
bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

ksp_options = [
	['ksp_type', 'cg'],
	['pc_type', 'hypre'],
	['pc_hypre_type', 'boomeramg'],
	['ksp_rtol', 1e-10],
	['ksp_atol', 1e-13],
	['ksp_max_it', 100],
]

adjoint_ksp_options = [
	['ksp_type', 'minres'],
	['pc_type', 'icc'],
	['pc_factor_levels', 0],
	['ksp_rtol', 1e-6],
	['ksp_atol', 1e-15],
]


y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config, ksp_options=ksp_options, adjoint_ksp_options=adjoint_ksp_options)
ocp.solve()
