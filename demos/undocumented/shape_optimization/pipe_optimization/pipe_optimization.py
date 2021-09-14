"""
Created on 15/06/2020, 16.11

@author: blauths
"""

from fenics import *

import cashocs


# load the config
config = cashocs.load_config("./config.ini")
# define the Reynold's number
Re = 4e2

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
J = Constant(1 / Re) * inner(grad(u), grad(u)) * dx

# define the optimization problem and solve it
optimization_problem = cashocs.ShapeOptimizationProblem(
    e, bcs, J, up, vq, boundaries, config
)
optimization_problem.solve()
