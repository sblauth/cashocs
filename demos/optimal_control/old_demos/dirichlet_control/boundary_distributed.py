"""
Created on 26.03.20, 07:21

@author: sebastian
"""

import configparser
from fenics import *
from cashocs import OptimalControlProblem, import_mesh
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = import_mesh(config.get('Mesh', 'mesh_file'))
V = FunctionSpace(mesh, 'CG', 1)
n = FacetNormal(mesh)
h = MaxCellEdgeLength(mesh)

bcs = []

y = Function(V)
u = Function(V)
u.vector()[:] = 0.0

p = Function(V)

eta = Constant(1e4)
e = inner(grad(y), grad(p))*dx - inner(grad(y), n)*p*ds - inner(grad(p), n)*(y - u)*ds + eta/h*(y - u)*p*ds - u*p*dx

lambd = 0.0
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)

J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*ds
# J = Constant(0.5)*pow(y - y_d, 2)*dx + Constant(0.5*lambd)*pow(u, 2)*dx

# control_constraints = [0, 10]
# control_constraints = [0.0, float('inf')]
# control_constraints = [float('-inf'), 0]
control_constraints = [float('-inf'), float('inf')]

optimization_problem = OptimalControlProblem(e, bcs, J, y, u, p, config, control_constraints=control_constraints)
optimization_problem.solve()



### Verify that we (approximately) have y=u on \Gamma
bc1 = DirichletBC(V, 1, boundaries, 1)
bc2 = DirichletBC(V, 1, boundaries, 2)
bc3 = DirichletBC(V, 1, boundaries, 3)
bc4 = DirichletBC(V, 1, boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

bdry_idx = Function(V)
[bc.apply(bdry_idx.vector()) for bc in bcs]
mask = np.where(bdry_idx.vector()[:] == 1)[0]

y_bdry = Function(V)
u_bdry = Function(V)
y_bdry.vector()[mask] = y.vector()[mask]
u_bdry.vector()[mask] = u.vector()[mask]

error_inf = np.max(np.abs(y_bdry.vector()[:] - u_bdry.vector()[:])) / np.max(np.abs(u_bdry.vector()[:])) * 100
error_l2 = np.sqrt(assemble((y - u)*(y - u)*ds)) / np.sqrt(assemble(u*u*ds)) * 100

print('Error regarding the (weak) imposition of the boundary values')
print('Error L^\infty: ' + format(error_inf, '.3e') + ' %')
print('Error L^2: ' + format(error_l2, '.3e') + ' %')
