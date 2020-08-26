"""
Created on 18.05.20, 16:10

@author: sebastian
"""

from fenics import *
from cashocs import OptimalControlProblem, regular_mesh
import numpy as np
import configparser



set_log_level(LogLevel.ERROR)
config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = regular_mesh(50)

CG1 = FunctionSpace(mesh, 'CG', 1)
DG1 = FunctionSpace(mesh, 'DG', 1)

bc1 = DirichletBC(CG1, Constant(0), boundaries, 1)
bc2 = DirichletBC(CG1, Constant(0), boundaries, 2)
bc3 = DirichletBC(CG1, Constant(0), boundaries, 3)
bc4 = DirichletBC(CG1, Constant(0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

y = Function(CG1)
u = Function(CG1)
p = Function(DG1)

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

lambd = 1e-6
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)

J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx

optimization_problem = OptimalControlProblem(e, bcs, J, y, u, p, config)
optimization_problem.solve()
