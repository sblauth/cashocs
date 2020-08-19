"""
Created on 13/08/2020, 13.15

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

e = inner(grad(y), grad(p))*dx + y*p*dx - u*p*ds

bcs = None

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*ds

scalar_product = TrialFunction(V)*TestFunction(V)*ds

ocp = cestrel.OptimalControlProblem(e, bcs, J, y, u, p, config, riesz_scalar_products=scalar_product)
ocp.solve()
