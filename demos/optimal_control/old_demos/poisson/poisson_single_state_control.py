"""
Created on 02/03/2020, 14.52

@author: blauths
"""

import configparser
from fenics import *
from adpack import OptimalControlProblem, MeshGen, regular_mesh
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

# mesh, subdomains, boundaries, dx, ds, dS = MeshGen(config.get('Mesh', 'mesh_file'))
mesh, subdomains, boundaries, dx, ds, dS = regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
u = Function(V)
u.vector()[:] = 0.0

p = Function(V)

# e = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx
e = inner(grad(y), grad(p))*dx - u*p*dx

bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

lambd = 1e-6
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)

J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx
# J = Constant(0.5)*pow(y - y_d, 2)*dx + Constant(0.5*lambd)*pow(u, 2)*dx

# control_constraints = [0, 10]
# control_constraints = [0.0, float('inf')]

# u_a = Function(V)
# u_b = Function(V)
# u_a.vector()[:] = 0.0
# u_b.vector()[:] = float('inf')
#
linear_a = Expression('50*(x[0]-1)', degree=1)
linear_b = Expression('50*x[0]', degree=1)
u_a = interpolate(linear_a, V)
u_b = interpolate(linear_b, V)

control_constraints = [u_a, u_b]
# control_constraints = [float('-inf'), 0]
# constraints = [float('-inf'), float('inf')]
#
# trial = TrialFunction(V)
# test = TestFunction(V)
# scalar_product = Constant(1e-5)*inner(grad(trial), grad(test))*dx + trial*test*dx

# bcs = bc1

optimization_problem = OptimalControlProblem(e, bcs, J, y, u, p, config)
optimization_problem.solve()
