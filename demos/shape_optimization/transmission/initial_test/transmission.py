"""
Created on 16/06/2020, 15.52

@author: blauths
"""

from fenics import *
from descendal import ShapeOptimizationProblem, import_mesh
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

sigma_out = 1.0
sigma_in = 10.0
rhs_g = 1.0

def generate_reference():
	mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/reference.xdmf')

	V = FunctionSpace(mesh, 'CG', 1)

	u = TrialFunction(V)
	v = TestFunction(V)

	a = sigma_out*inner(grad(u), grad(v))*dx(1) + sigma_in*inner(grad(u), grad(v))*dx(2)
	L = Constant(rhs_g)*v*ds(1)

	bcs = DirichletBC(V, Constant(0), boundaries, 2)

	reference = Function(V)
	solve(a==L, reference, bcs)

	return reference

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/mesh.xdmf')

V = FunctionSpace(mesh, 'CG', 1)

reference = Function(V)
reference.vector()[:] = interpolate(generate_reference(), V).vector()[:]

bcs = DirichletBC(V, Constant(0), boundaries, 2)

u = Function(V)
p = Function(V)

e = sigma_out*inner(grad(u), grad(p))*dx(1) + sigma_in*inner(grad(u), grad(p))*dx(2) - Constant(rhs_g)*p*ds(1)

J = pow(u - reference, 2)*ds(1)

optimization_problem = ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
# optimization_problem.state_problem.solve()
optimization_problem.solve()

DG0 = FunctionSpace(mesh, 'DG', 0)
post = Function(DG0)
a = TrialFunction(DG0)*TestFunction(DG0)*dx
L = Constant(1)*TestFunction(DG0)*dx(1) + Constant(2)*TestFunction(DG0)*dx(2)

solve(a==L, post)

post_file = File('post.pvd')
post_file << post
