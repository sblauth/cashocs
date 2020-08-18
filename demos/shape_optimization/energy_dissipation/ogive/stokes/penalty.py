"""
Created on 23/06/2020, 15.48

@author: blauths
"""

from fenics import *
from descendal import ShapeOptimizationProblem, import_mesh
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/mesh.xdmf')

x = SpatialCoordinate(mesh)
volume_initial = 4*9 - assemble(1*dx)
barycenter_x_initial = (1/2*(6**2 - 3**2)*4 - assemble(x[0]*dx)) / volume_initial
barycenter_y_initial = (1/2*(2**2 - 2**2)*9 - assemble(x[1]*dx)) / volume_initial

v_elem = VectorElement('CG', mesh.ufl_cell(), 2)
p_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
space = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

# v_in = Expression(('cos(1.0/4.0*pi*x[1])', '0.0'), degree=2)
v_in = Expression(('-1.0/4.0*(x[1] - 2.0)*(x[1] + 2.0)', '0.0'), degree=2)
bc_in = DirichletBC(space.sub(0), v_in, boundaries, 1)
bc_wall = DirichletBC(space.sub(0), Constant((0, 0)), boundaries, 2)
bc_gamma = DirichletBC(space.sub(0), Constant((0, 0)), boundaries, 4)
bcs = [bc_in, bc_wall, bc_gamma]


up = Function(space)
u, p = split(up)
vq = Function(space)
v, q = split(vq)

e = inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx

J = inner(grad(u), grad(u))*dx


regularizations = [1e2, 1e4, 1e5, 1e6, 1e7]
i_vol = assemble(1*dx)
i_bc_x = assemble(x[0]*dx) / i_vol
i_bc_y = assemble(x[1]*dx) / i_vol

config.set('Regularization', 'target_barycenter', '[' + str(i_bc_x) +', ' + str(i_bc_y) + ', 0]')
config.set('Regularization', 'target_volume', str(i_vol))
config.set('Regularization', 'use_initial_volume', 'false')
config.set('Regularization', 'use_initial_barycenter', 'false')

for reg in regularizations:

	config.set('Regularization', 'factor_target_volume', str(reg))
	config.set('Regularization', 'factor_barycenter', str(reg))

	optimization_problem = ShapeOptimizationProblem(e, bcs, J, up, vq, boundaries, config)
	optimization_problem.solve()
# optimization_problem.state_problem.solve()

u, p = up.split(True)

volume = 4*9 - assemble(1*dx)
barycenter_x = (1/2*(6**2 - 3**2)*4 - assemble(x[0]*dx)) / volume
barycenter_y = (1/2*(2**2 - 2**2)*9 - assemble(x[1]*dx)) / volume
