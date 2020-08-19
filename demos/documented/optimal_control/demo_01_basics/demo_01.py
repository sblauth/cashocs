"""
Created on 13/08/2020, 10.54

@author: blauths
"""

from fenics import *
import cestrel



set_log_level(LogLevel.CRITICAL)
config = cestrel.create_config('config.ini')

mesh, subdomains, boundaries, dx, ds, dS = cestrel.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
p = Function(V)
u = Function(V)

e = inner(grad(y), grad(p))*dx - u*p*dx

bcs = cestrel.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

ocp = cestrel.OptimalControlProblem(e, bcs, J, y, u, p, config)
ocp.solve()
