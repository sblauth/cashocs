"""
Created on 03/03/2020, 11.30

@author: blauths
"""

import configparser
from fenics import *
from caospy import OptimalControlProblem, MeshGen
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = MeshGen(config.get('Mesh', 'mesh_file'))
V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'CG', 2)

y = Function(V)
z = Function(W)
u = Function(V)
v = Function(V)

p = Function(V)
q = Function(W)

e1 = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx - v*p*ds
e2 = inner(grad(z), grad(q))*dx + y*q*dx - u*q*dx

bc1 = DirichletBC(W, Constant(0), boundaries, 1)
bc2 = DirichletBC(W, Constant(0), boundaries, 2)
bc3 = DirichletBC(W, Constant(0), boundaries, 3)
bc4 = DirichletBC(W, Constant(0), boundaries, 4)

bcs1 = []
bcs2 = [bc1, bc2, bc3, bc4]
bcs_list = [bcs1, bcs2]


lambd = 1e-4
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
z_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)

### Distributed control and observation
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx + Constant(0.5*lambd)*u*u*dx

scalar_products = [TrialFunction(V)*TestFunction(V)*dx, TrialFunction(V)*TestFunction(V)*ds]

optimization_problem = OptimalControlProblem([e1, e2], bcs_list, J, [y, z], [u, v], [p, q], config, scalar_products)
optimization_problem.solve()
