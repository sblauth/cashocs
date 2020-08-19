"""
Created on 15/06/2020, 16.11

@author: blauths
"""

from fenics import *
from cestrel import ShapeOptimizationProblem, import_mesh
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

Re = 6e2

mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/mesh.xdmf')

x = SpatialCoordinate(mesh)
volume_initial = assemble(1*dx)

v_elem = VectorElement('CG', mesh.ufl_cell(), 2, dim=2)
p_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
space = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

v_in = Expression(('-4*(x[1] - 1)*(x[1] + 0)', '0.0'), degree=2)
bc_in = DirichletBC(space.sub(0), v_in, boundaries, 1)
bc_wall = DirichletBC(space.sub(0), Constant((0, 0)), boundaries, 2)
bc_gamma = DirichletBC(space.sub(0), Constant((0, 0)), boundaries, 4)
bcs = [bc_in, bc_wall, bc_gamma]


up = Function(space)
u, p = split(up)
vq = Function(space)
v, q = split(vq)

e = Constant(1/Re)*inner(grad(u), grad(v))*dx + inner(grad(u)*u, v)*dx - p*div(v)*dx - q*div(u)*dx

J = Constant(1/Re)*inner(grad(u), grad(u))*dx

optimization_problem = ShapeOptimizationProblem(e, bcs, J, up, vq, boundaries, config)
optimization_problem.solve()
# optimization_problem.state_problem.solve()

u, p = up.split(True)

volume = assemble(1*dx)
