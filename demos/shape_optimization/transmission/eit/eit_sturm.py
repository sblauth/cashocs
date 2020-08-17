"""
Created on 16/06/2020, 15.52

@author: blauths
"""

from fenics import *
from adoptpy import ShapeOptimizationProblem, import_mesh
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

sigma_out = 1e0
sigma_in = 1e1

def generate_references():
	mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/reference.xdmf')

	cg_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
	r_elem = FiniteElement('R', mesh.ufl_cell(), 0)
	V = FunctionSpace(mesh, MixedElement([cg_elem, r_elem]))

	u, c = TrialFunctions(V)
	v, d = TestFunctions(V)

	a = sigma_out*inner(grad(u), grad(v))*dx(1) + sigma_in*inner(grad(u), grad(v))*dx(2) + u*d*dx + v*c*dx
	L1  = Constant(1)*v*(ds(3) + ds(4)) + Constant(-1)*v*(ds(1) + ds(2))
	L2  = Constant(1)*v*(ds(3) + ds(2)) + Constant(-1)*v*(ds(1) + ds(4))
	L3  = Constant(1)*v*(ds(3) + ds(1)) + Constant(-1)*v*(ds(2) + ds(4))

	reference1 = Function(V)
	reference2 = Function(V)
	reference3 = Function(V)
	solve(a==L1, reference1)
	solve(a==L2, reference2)
	solve(a==L3, reference3)

	ref1, _ = reference1.split(True)
	ref2, _ = reference2.split(True)
	ref3, _ = reference3.split(True)

	return [ref1, ref2, ref3]


config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/mesh.xdmf')
cg_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([cg_elem, cg_elem]))

references = generate_references()
# ref1 = Function(V)
# ref2 = Function(V)
# ref3 = Function(V)
# ref1.vector()[:] = interpolate(references[0], V).vector()[:]
# ref2.vector()[:] = interpolate(references[1], V).vector()[:]
# ref3.vector()[:] = interpolate(references[2], V).vector()[:]

bc1_top = DirichletBC(V.sub(0), references[0], boundaries, 2)
bc1_bot = DirichletBC(V.sub(0), references[0], boundaries, 1)
bc1_left = DirichletBC(V.sub(1), references[0], boundaries, 3)
bc1_right = DirichletBC(V.sub(1), references[0], boundaries, 4)
bcs1 = [bc1_top, bc1_bot, bc1_left, bc1_right]

bc2_top = DirichletBC(V.sub(0), references[1], boundaries, 2)
bc2_bot = DirichletBC(V.sub(0), references[1], boundaries, 1)
bc2_left = DirichletBC(V.sub(1), references[1], boundaries, 3)
bc2_right = DirichletBC(V.sub(1), references[1], boundaries, 4)
bcs2 = [bc2_top, bc2_bot, bc2_left, bc2_right]

bc3_top = DirichletBC(V.sub(0), references[2], boundaries, 2)
bc3_bot = DirichletBC(V.sub(0), references[2], boundaries, 1)
bc3_left = DirichletBC(V.sub(1), references[2], boundaries, 3)
bc3_right = DirichletBC(V.sub(1), references[2], boundaries, 4)
bcs3 = [bc3_top, bc3_bot, bc3_left, bc3_right]

bcs = [bcs1, bcs2, bcs3]

u1 = Function(V)
p1 = Function(V)
e1 = sigma_out*inner(grad(u1[0]), grad(p1[0]))*dx(1) + sigma_in*inner(grad(u1[0]), grad(p1[0]))*dx(2) - Constant(1)*p1[0]*(ds(3) + ds(4)) \
	+ sigma_out*inner(grad(u1[1]), grad(p1[1]))*dx(1) + sigma_in*inner(grad(u1[1]), grad(p1[1]))*dx(2) - Constant(-1)*p1[1]*(ds(1) + ds(2))

u2 = Function(V)
p2 = Function(V)
e2 = sigma_out*inner(grad(u2[0]), grad(p2[0]))*dx(1) + sigma_in*inner(grad(u2[0]), grad(p2[0]))*dx(2) - Constant(1)*p2[0]*ds(3) - Constant(-1)*p2[0]*ds(4) \
	+ sigma_out*inner(grad(u2[1]), grad(p2[1]))*dx(1) + sigma_in*inner(grad(u2[1]), grad(p2[1]))*dx(2) - Constant(1)*p2[1]*ds(2) - Constant(-1)*p2[1]*ds(1)

u3 = Function(V)
p3 = Function(V)
e3 = sigma_out*inner(grad(u3[0]), grad(p3[0]))*dx(1) + sigma_in*inner(grad(u3[0]), grad(p3[0]))*dx(2) - Constant(1)*p3[0]*ds(3) - Constant(-1)*p3[0]*ds(4) \
	+ sigma_out*inner(grad(u3[1]), grad(p3[1]))*dx(1) + sigma_in*inner(grad(u3[1]), grad(p3[1]))*dx(2) - Constant(1)*p3[1]*ds(1) - Constant(-1)*p3[1]*ds(2)

e = [e1, e2, e3]
u = [u1, u2, u3]
p = [p1, p2, p3]

mu1 = Expression('val', degree=0, val=1.0)
mu2 = Expression('val', degree=0, val=1.0)
mu3 = Expression('val', degree=0, val=1.0)

### Domain
J1 = mu1*Constant(0.5)*pow(u1[0] - u1[1], 2)*dx
J2 = mu2*Constant(0.5)*pow(u2[0] - u2[1], 2)*dx
J3 = mu3*Constant(0.5)*pow(u3[0] - u3[1], 2)*dx

### boundary
# J1 = mu1*Constant(0.5)*pow(u1[0] - u1[1], 2)*ds
# J2 = mu2*Constant(0.5)*pow(u2[0] - u2[1], 2)*ds
# J3 = mu3*Constant(0.5)*pow(u3[0] - u3[1], 2)*ds

### boundary vs measurement
# J1 = mu1*(Constant(0.5)*pow(u1[0] - references[0], 2)*ds + Constant(0.5)*pow(u1[1] - references[0], 2)*ds)
# J2 = mu2*(Constant(0.5)*pow(u2[0] - references[1], 2)*ds + Constant(0.5)*pow(u2[1] - references[1], 2)*ds)
# J3 = mu3*(Constant(0.5)*pow(u3[0] - references[2], 2)*ds + Constant(0.5)*pow(u3[1] - references[2], 2)*ds)

J = J1 + J2 + J3

optimization_problem = ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
optimization_problem.state_problem.solve()

mu1.val = 1/assemble(J1)
mu2.val = 1/assemble(J2)
mu3.val = 1/assemble(J3)

optimization_problem.solve()


# ### post processing

DG0 = FunctionSpace(mesh, 'DG', 0)
post = Function(DG0)
a = TrialFunction(DG0)*TestFunction(DG0)*dx
L = Constant(1)*TestFunction(DG0)*dx(1) + Constant(2)*TestFunction(DG0)*dx(2)

solve(a==L, post)

post_file = File('post.pvd')
post_file << post
