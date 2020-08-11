"""
Created on 23.03.20, 15:15

@author: sebastian
"""

import configparser
from fenics import *
from adpack import OptimalControlProblem, MeshGen
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = MeshGen(config.get('Mesh', 'mesh_file'))
V = FunctionSpace(mesh, 'CG', 1)
R = FunctionSpace(mesh, 'R', 0)
n = FacetNormal(mesh)
h = MaxCellEdgeLength(mesh)

bcs = []

y = Function(V)
u = Function(V)
c = Function(R)
# c.vector()[:] = 10

p = Function(V)

eta = Constant(1e3)

ds_rest = ds(2) + ds(3) + ds(4)

e = inner(grad(y), grad(p))*dx \
	- inner(grad(y), n)*p*ds - inner(grad(p), n)*(y - u)*ds_rest - inner(grad(p), n)*(y - c)*ds(1) \
	+ eta/h*(y - u)*p*ds_rest + eta/h*(y - c)*p*ds(1) \
	- Constant(1)*p*dx

lambd = 0.0
gamma = Constant(5)
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)

J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*ds + gamma*(u - c)*(u - c)*ds(1)
# J = Constant(0.5)*pow(y - y_d, 2)*dx + Constant(0.5*lambd)*pow(u, 2)*dx

scalar_products = [TrialFunction(V)*TestFunction(V)*ds, TrialFunction(R)*TestFunction(R)*dx]

optimization_problem = OptimalControlProblem(e, bcs, J, y, [u, c], p, config, riesz_scalar_products=scalar_products)
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

print('Error regarding the (weak) imposition of the boundary values:')
print('Error L^\infty: ' + format(error_inf, '.3e') + ' %')
print('Error L^2: ' + format(error_l2, '.3e') + ' %')


### Test if u = c on ds(1)
bc1 = DirichletBC(V, -1, boundaries, 1)
bdry_idx = Function(V)
bc1.apply(bdry_idx.vector())
mask = np.where(bdry_idx.vector()[:] == -1)[0]

error_inf = np.max(np.abs(u.vector()[mask] - c.vector()[:])) / np.max(np.abs(c.vector()[:])) * 100
error_l2 = np.sqrt(assemble((u - c)*(u - c)*ds(1))) / np.sqrt(assemble(c*c*ds(1))) * 100
print('')
print('Error of u=c on ds(1): ')
print('Error L^\infty: ' + format(error_inf, '.3e') + ' %')
print('Error L^2: ' + format(error_l2, '.3e') + ' %')
