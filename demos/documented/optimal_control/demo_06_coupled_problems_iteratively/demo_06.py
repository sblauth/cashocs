"""
Created on 13/08/2020, 14.35

@author: blauths
"""

from fenics import *
import cestrel



set_log_level(LogLevel.CRITICAL)
config = cestrel.create_config('config.ini')

mesh, subdomains, boundaries, dx, ds, dS = cestrel.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
z = Function(V)
p = Function(V)
q = Function(V)
states = [y, z]
adjoints = [p, q]

u = Function(V)
v = Function(V)
controls = [u, v]

e1 = inner(grad(y), grad(p))*dx + z*p*dx - u*p*dx
e2 = inner(grad(z), grad(q))*dx + y*q*dx - v*q*dx
e = [e1, e2]

bcs1 = cestrel.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs2 = cestrel.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs = [bcs1, bcs2]

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
alpha = 1e-6
beta = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
	+ Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

optimization_problem = cestrel.OptimalControlProblem(e, bcs, J, states, controls, adjoints, config)
optimization_problem.solve()
