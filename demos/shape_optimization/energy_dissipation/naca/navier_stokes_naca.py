"""
Created on 15/06/2020, 16.11

@author: blauths
"""

from fenics import *
from adpack.geometry import MeshGen
from adpack import ShapeOptimizationProblem
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = MeshGen('./mesh/mesh.xdmf')

Re = 1e2

x = SpatialCoordinate(mesh)
volume_initial = 4.5*1 - assemble(1*dx)
barycenter_x_initial = (1/2*(3**2 - 1.5**2)*1 - assemble(x[0]*dx)) / volume_initial
barycenter_y_initial = (1/2*(0.5**2 - 0.5**2)*4.5 - assemble(x[1]*dx)) / volume_initial

v_elem = VectorElement('CG', mesh.ufl_cell(), 2)
p_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

v_in = Expression(('-4.0*(x[1] - 0.5)*(x[1] + 0.5)', '0.0'), degree=2)
bc_in = DirichletBC(V.sub(0), v_in, boundaries, 1)
bc_wall = DirichletBC(V.sub(0), Constant((0, 0)), boundaries, 2)
bc_gamma = DirichletBC(V.sub(0), Constant((0, 0)), boundaries, 4)
bcs = [bc_in, bc_wall, bc_gamma]


up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

e = Constant(1/Re)*inner(grad(u), grad(v))*dx + inner(grad(u)*u, v)*dx - p*div(v)*dx - q*div(u)*dx

J = Constant(1/Re)*inner(grad(u), grad(u))*dx

optimization_problem = ShapeOptimizationProblem(e, bcs, J, up, vq, boundaries, config)
optimization_problem.solve()

u, p = up.split(True)

volume = 4.5*1 - assemble(1*dx)
barycenter_x = (1/2*(3**2 - 1.5**2)*1 - assemble(x[0]*dx)) / volume
barycenter_y = (1/2*(0.5**2 - 0.5**2)*4.5 - assemble(x[1]*dx)) / volume
