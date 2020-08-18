"""
Created on 13/08/2020, 14.27

@author: blauths
"""

from fenics import *
import descendal



set_log_level(LogLevel.CRITICAL)
config = descendal.create_config('config.ini')

mesh, subdomains, boundaries, dx, ds, dS = descendal.regular_mesh(50)

# Create different spaces for state and adjoint variables
V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'DG', 1)

y = Function(V)
p = Function(W)
u = Function(V)

# Set up a discontinuous Galerkin method (SIPG) (needed for the adjoint system,
# reduces to the classical Galerkin method for CG elements)
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2
flux = 1/2*(inner(grad(u)('+') + grad(u)('-'), n('+')))
alpha = 1e3
gamma = 1e3
e = dot(grad(p), grad(y))*dx \
	- dot(avg(grad(p)), jump(y, n))*dS \
	- dot(jump(p, n), avg(grad(y)))*dS \
	+ Constant(alpha)/h_avg*dot(jump(p, n), jump(y, n))*dS \
	- dot(grad(p), y*n)*ds \
	- dot(p*n, grad(y))*ds \
	+ (Constant(gamma)/h)*p*y*ds - u*p*dx

bcs = descendal.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

lambd = 1e-6
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)

J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx

optimization_problem = descendal.OptimalControlProblem(e, bcs, J, y, u, p, config)
optimization_problem.solve()
