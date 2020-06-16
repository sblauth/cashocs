"""
Created on 03.04.20, 10:28

@author: sebastian
"""

import configparser
from fenics import *
from adpack import OptimalControlProblem, MeshGen, regular_mesh
from ufl import Max
import numpy as np



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

# mesh, subdomains, boundaries, dx, ds, dS = MeshGen(config.get('Mesh', 'mesh_file'))
mesh, subdomains, boundaries, dx, ds, dS = regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
u = Function(V)
u.vector()[:] = 0.0

shift = Function(V)
shift.vector()[:] = 0.0
indicator = Function(V)

p = Function(V)

# e = inner(grad(y), grad(p))*dx + Constant(1.0)*y*p*dx - u*p*dx
e = inner(grad(y), grad(p))*dx - u*p*dx

bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

lambd = 1e-3

y_b = 1e-1
y_d = Expression('sin(2*pi*x[0]*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
# y_d = Expression('sin(2*pi*x[0])', degree=1)
control_constraints = [float('-inf'), float('inf')]

J_init = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx
optimization_problem = OptimalControlProblem(e, bcs, J_init, y, u, p, config, control_constraints=control_constraints)
optimization_problem.solve()


class Sets:
	def __init__(self):
		self.idx_active_prev = np.array([])
		self.idx_active_pprev = np.array([])
		self.initialized = False
		self.converged = False

	def compute(self, gamma):

		self.idx_active = (shift.vector()[:] + gamma*(y.vector()[:] - y_b) > 0).nonzero()[0]
		self.idx_active.sort()
		print('No active idcs: ' + str(len(self.idx_active)))
		self.idx_inactive = np.setdiff1d(np.arange(V.dim()), self.idx_active)

		if self.initialized:
			if np.array_equal(self.idx_active, self.idx_active_prev) or np.array_equal(self.idx_active, self.idx_active_pprev):
				self.converged = True

		self.idx_active_pprev = self.idx_active_prev
		self.idx_active_prev = self.idx_active

		self.initialized = True
		indicator.vector()[:] = 0.0
		indicator.vector()[self.idx_active] = 1.0


# gammas = [pow(10, i) for i in np.arange(0, 10, 3)]
gammas = [pow(10, i) for i in np.arange(1, 6, 1)]
# gammas = [1e4 for i in range(10)]

# gamma = 1e3
for gamma in gammas:

	sets = Sets()
	for kk in range(25):
		# if kk==0:
		# 	config['OptimizationRoutine']['tolerance'] = 1e-2
		sets.compute(gamma)
		if sets.converged:
			print('PDAS converged')
			break
		J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx + Constant(1/(2*gamma))*pow(shift + Constant(gamma)*(y - y_b), 2)*indicator*dx


		# J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*lambd)*u*u*dx + Constant(1/(2*gamma))*pow(Max(0, shift + Constant(gamma)*(y - y_b)), 2)*indicator*dx

		optimization_problem = OptimalControlProblem(e, bcs, J, y, u, p, config, control_constraints=control_constraints)
		optimization_problem.solve()

	# if uncommented, gives augmented lagrangian approach
	# shift.vector()[:] += np.maximum(0, shift.vector()[:] + gamma*(y.vector()[:] - y_b))
	# shift.vector()[sets.idx_inactive] = 0.0

	if not sets.converged:
		print('PDAS did not converge')
		break

	print('')

# y_file = File('y.pvd')
# u_file = File('u.pvd')
# y_file << y
# u_file << u
