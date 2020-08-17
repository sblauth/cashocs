"""
Created on 24/02/2020, 08.39

@author: blauths
"""

import configparser
from fenics import *
from adoptpy import OptimalControlProblem, import_mesh
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = import_mesh(config.get('Mesh', 'mesh_file'))
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
u = Function(V)
u.vector()[:] = 0.0
v = Function(V)
v.vector()[:] = 0.0

p = Function(V)

e = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx - v*p*ds
# e = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx + (y - v)*p*dx
# e = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*ds

bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
# bcs = [bc1, bc2, bc3, bc4]
bcs = []


lambd = 1e-4
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)

### Distributed control and observation
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx + Constant(0.5*lambd)*v*v*ds

### Boundary control and observation
# J = Constant(0.5)*(y - y_d)*(y - y_d)*ds + Constant(0.5*lambd)*u*u*ds

### L^1 sparse control
# J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(lambd)*abs(u)*dx

scalar_products = [TrialFunction(V)*TestFunction(V)*dx, TrialFunction(V)*TestFunction(V)*ds]

optimization_problem = OptimalControlProblem([e], [bcs], J, [y], [u, v], [p], config, scalar_products)
optimization_problem.solve()
