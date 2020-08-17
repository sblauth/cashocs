"""
Created on 13/08/2020, 14.35

@author: blauths
"""

from fenics import *
import adoptpy



set_log_level(LogLevel.CRITICAL)
config = adoptpy.create_config('config.ini')

mesh, subdomains, boundaries, dx, ds, dS = adoptpy.regular_mesh(50)

elem_1 = FiniteElement('CG', mesh.ufl_cell(), 1)
elem_2 = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([elem_1, elem_2]))

U = FunctionSpace(mesh, 'CG', 1)

state = Function(V)
adjoint = Function(V)
y, z = split(state)
p, q = split(adjoint)

u = Function(U)
v = Function(U)
controls = [u, v]

e1 = inner(grad(y), grad(p))*dx + z*p*dx - u*p*dx
e2 = inner(grad(z), grad(q))*dx + y*q*dx - v*q*dx
e = e1 + e2

bcs1 = adoptpy.create_bcs_list(V.sub(0), Constant(0), boundaries, [1, 2, 3, 4])
bcs2 = adoptpy.create_bcs_list(V.sub(1), Constant(0), boundaries, [1, 2, 3, 4])
bcs = bcs1 + bcs2

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
alpha = 1e-6
beta = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
	+ Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

optimization_problem = adoptpy.OptimalControlProblem(e, bcs, J, state, controls, adjoint, config)
optimization_problem.solve()
