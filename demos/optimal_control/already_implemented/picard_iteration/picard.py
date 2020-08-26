"""
Created on 30.03.20, 09:49

@author: sebastian
"""

import configparser
from fenics import *
from cashocs import OptimalControlProblem, import_mesh
import cashocs
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

# mesh, subdomains, boundaries, dx, ds, dS = import_mesh('../mesh/mesh.xdmf')
mesh, subdomains, boundaries, dx, ds, dS = import_mesh(config)
# mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
z = Function(V)
u = Function(V)
v = Function(V)

u.vector()[:] = 1.0
v.vector()[:] = 1.0

p = Function(V)
q = Function(V)

e1 = inner(grad(y), grad(p))*dx + y*p*dx - u*p*dx - z*p*dx
e2 = inner(grad(z), grad(q))*dx - v*q*dx - y*q*dx

bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs1 = [bc1, bc2, bc3, bc4]
bcs2 = [bc1, bc2, bc3, bc4]

lambd = 1e-2
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)

J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - y_d)*(z - y_d)*dx + Constant(0.5*lambd)*u*u*dx
# J = Constant(0.5)*pow(y - y_d, 2)*dx + Constant(0.5*lambd)*pow(u, 2)*dx

# control_constraints = [0, 10]
# control_constraints = [0.0, float('inf')]
# control_constraints = [float('-inf'), 0]
control_constraints = [float('-inf'), float('inf')]


optimization_problem = OptimalControlProblem([e1, e2], [bcs1, bcs2], J, [y, z], [u, v], [p, q], config)

optimization_problem.solve()
