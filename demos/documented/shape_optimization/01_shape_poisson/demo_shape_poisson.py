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

"""The documentation of this program is found under ./doc_shape_poisson.md

"""

from fenics import *

import cashocs



config = cashocs.create_config('./config.ini')

meshlevel = 25
degree = 1
dim = 2
mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
dx = Measure('dx', mesh)
boundary = CompiledSubDomain('on_boundary')
boundaries = MeshFunction('size_t', mesh, dim=1)
boundary.mark(boundaries, 1)
ds = Measure('ds', mesh, subdomain_data=boundaries)

V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
p = Function(V)

x = SpatialCoordinate(mesh)
f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

e = inner(grad(u), grad(p))*dx - f*p*dx
bcs = DirichletBC(V, Constant(0), boundaries, 1)

J = u*dx

sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
sop.solve()
