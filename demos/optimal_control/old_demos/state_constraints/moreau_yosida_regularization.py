"""
Created on 03.04.20, 09:46

@author: sebastian
"""

import configparser
from fenics import *
from cashocs import OptimalControlProblem, import_mesh, regular_mesh
from ufl import Max
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

# mesh, subdomains, boundaries, dx, ds, dS = import_mesh(config.get('Mesh', 'mesh_file'))
mesh, subdomains, boundaries, dx, ds, dS = regular_mesh(25)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
u = Function(V)
u.vector()[:] = 0.0
u_init = Function(V)

p = Function(V)

# e = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx
e = inner(grad(y), grad(p))*dx - u*p*dx

bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

lambd = 1e-3
shift = 0.0
y_b = 1e-1
y_d = Expression('sin(2*pi*x[0]*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)
control_constraints = [float('-inf'), float('inf')]

J_init = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx
optimization_problem = OptimalControlProblem(e, bcs, J_init, y, u, p, config, control_constraints=control_constraints)
optimization_problem.solve()
# gammas = [pow(10, i) for i in np.arange(1, 4, 1)]
gammas = [pow(10, i) for i in np.arange(1, 10, 3)]


for gamma in gammas:

	J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx + Constant(1/(2*gamma))*pow(Max(0, Constant(shift) + Constant(gamma)*(y - y_b)), 2)*dx

	optimization_problem = OptimalControlProblem(e, bcs, J, y, u, p, config, control_constraints=control_constraints)
	optimization_problem.solve()


y_file = File('y.pvd')
u_file = File('u.pvd')
y_file << y
u_file << u
