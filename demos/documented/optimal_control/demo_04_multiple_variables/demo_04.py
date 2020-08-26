"""
Created on 13/08/2020, 13.12

@author: blauths
"""

from fenics import *
import cashocs



set_log_level(LogLevel.CRITICAL)
config = cashocs.create_config('config.ini')

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y1 = Function(V)
y2 = Function(V)
p1 = Function(V)
p2 = Function(V)
u = Function(V)
v = Function(V)

y = [y1, y2]
p = [p1, p2]
controls = [u, v]

e1 = inner(grad(y1), grad(p1))*dx - u*p1*dx
e2 = inner(grad(y2), grad(p2))*dx - (y1 + v)*p2*dx

pdes = [e1, e2]

bcs1 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs2 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

bcs_list = [bcs1, bcs2]

y1_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
y2_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
alpha = 1e-6
beta = 1e-4
J = Constant(0.5)*(y1 - y1_d)*(y1 - y1_d)*dx + Constant(0.5)*(y2 - y2_d)*(y2 - y2_d)*dx \
	+ Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

optimization_problem = cashocs.OptimalControlProblem(pdes, bcs_list, J, y, controls, p, config)
optimization_problem.solve()
