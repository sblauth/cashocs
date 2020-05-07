"""
Created on 03/03/2020, 11.10

@author: blauths
"""

import configparser
from fenics import *
from adpack import OptimizationProblem, MeshGen
import numpy as np
import time



start_time = time.time()
set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = MeshGen(config.get('Mesh', 'mesh_file'))
V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'CG', 2)

y = Function(V)
z = Function(W)
u = Function(V)
u.vector()[:] = 0.0

p = Function(V)
q = Function(W)

e1 = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx
e2 = inner(grad(z), grad(q))*dx - u*q*dx

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

optimization_problem = OptimizationProblem([e1, e2], bcs_list, [dx], J, [y, z], [u], [p, q], config)
optimization_problem.solve()
end_time = time.time()
print('Ellapsed time: ' + str(end_time - start_time) + ' s')
