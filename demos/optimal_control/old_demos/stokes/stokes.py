"""
Created on 24/02/2020, 11.47

@author: blauths
"""

import configparser
from fenics import *
from cestrel import OptimalControlProblem, import_mesh
import cestrel
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = import_mesh(config.get('Mesh', 'mesh_file'))
v_elem = VectorElement('CG', mesh.ufl_cell(), 2)
p_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))
U = V.sub(0).collapse()

up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

control = Function(U)

e = inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx - inner(control, v)*dx

def pressure_point(x, on_boundary):
	return on_boundary and near(x[0], 0) and near(x[1], 0)

bc1 = DirichletBC(V.sub(0), Constant((0, 0)), boundaries, 1)
bc2 = DirichletBC(V.sub(0), Constant((0, 0)), boundaries, 2)
bc3 = DirichletBC(V.sub(0), Constant((0, 0)), boundaries, 3)
velo_top = Expression(('4*x[0]*(1-x[0])', '0.0'), degree=2)
bc4 = DirichletBC(V.sub(0), velo_top, boundaries, 4)
bc_p = DirichletBC(V.sub(1), Constant(0), pressure_point, method='pointwise')

bcs = [bc1, bc2, bc3, bc4]


lambd = 1e-5

u_d = Expression(('sqrt(pow(x[0], 2) + pow(x[1], 2))*cos(2*pi*x[1])', '-sqrt(pow(x[0], 2) + pow(x[1], 2))*sin(2*pi*x[0])'), degree=2)
J = Constant(0.5)*inner(u - u_d, u - u_d)*dx + Constant(0.5*lambd)*inner(control, control)*dx

optimization_problem = OptimalControlProblem(e, bcs, J, up, control, vq, config)
optimization_problem.solve()
# optimization_problem.state_problem.solve()

u, p = up.split()
v, q = vq.split()

