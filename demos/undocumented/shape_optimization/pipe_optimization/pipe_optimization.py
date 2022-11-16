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

from fenics import *

import cashocs

# load the config
config = cashocs.load_config("./config.ini")
# define the Reynold's number
Re = 350.0

# import the mesh and geometry
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh("./mesh/mesh.xdmf")

# set up the function space
v_elem = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
space = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

# define boundary conditions
v_in = Expression(("-6*(x[1] - 1)*(x[1] + 0)", "0.0"), degree=2)
bc_in = DirichletBC(space.sub(0), v_in, boundaries, 1)
bc_wall = DirichletBC(space.sub(0), Constant((0, 0)), boundaries, 2)
bc_gamma = DirichletBC(space.sub(0), Constant((0, 0)), boundaries, 4)
bcs = [bc_in, bc_wall, bc_gamma]

# set up state and adjoint variables
up = Function(space)
u, p = split(up)
vq = Function(space)
v, q = split(vq)

# define the PDE constraint
e = (
    Constant(1 / Re) * inner(grad(u), grad(v)) * dx
    + inner(grad(u) * u, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
)

# set up the cost functional
J = cashocs.IntegralFunctional(Constant(1 / Re) * inner(grad(u), grad(u)) * dx)

# define the optimization problem and solve it
optimization_problem = cashocs.ShapeOptimizationProblem(
    e, bcs, J, up, vq, boundaries, config=config
)
optimization_problem.solve()
