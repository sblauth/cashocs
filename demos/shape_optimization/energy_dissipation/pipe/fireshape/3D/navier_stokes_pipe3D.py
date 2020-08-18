"""
Created on 16/06/2020, 11.14

@author: blauths
"""

from fenics import *
from descendal import ShapeOptimizationProblem, import_mesh
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

Re = 8e2

mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/mesh.xdmf')

x = SpatialCoordinate(mesh)
volume_initial = assemble(1*dx)

v_elem = VectorElement('CG', mesh.ufl_cell(), 2)
p_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

v_in = Expression(('0.0', '0.0', '1 - 4*(pow(x[0], 2) + pow(x[1], 2))'), degree=2)
bc_in = DirichletBC(V.sub(0), v_in, boundaries, 1)
bc_wall = DirichletBC(V.sub(0), Constant((0, 0, 0)), boundaries, 2)
bc_gamma = DirichletBC(V.sub(0), Constant((0, 0, 0)), boundaries, 4)
bcs = [bc_in, bc_wall, bc_gamma]


up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

e = Constant(1/Re)*inner(grad(u), grad(v))*dx + inner(grad(u)*u, v)*dx - p*div(v)*dx - q*div(u)*dx

J = Constant(1/Re)*inner(grad(u), grad(u))*dx

optimization_problem = ShapeOptimizationProblem(e, bcs, J, up, vq, boundaries, config)
optimization_problem.solve()
# optimization_problem.state_problem.solve()

u, p = up.split(True)

volume = assemble(1*dx)
