"""
Created on 20.04.20, 08:49

@author: sebastian
"""

import configparser
from fenics import *
from caospy import OptimalControlProblem, regular_mesh
from caospy.utilities import summation
import numpy as np
import time



start_time = time.time()
set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = regular_mesh(50)

dt = 1 / 40
t_start = dt
t_end = 1.0
t_array = np.linspace(t_start, t_end, int(1/dt))

V = FunctionSpace(mesh, 'CG', 1)

bc1 = DirichletBC(V, Constant(0.0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0.0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0.0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0.0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

states = [Function(V) for i in range(len(t_array))]
controls = [Function(V) for i in range(len(t_array))]
adjoints = [Function(V) for i in range(len(t_array))]
y_d = []
bcs_list = [bcs for i in range(len(t_array))]
control_measure = [dx for i in range(len(t_array))]
e = []
J_summands = []

lambd = 1e-6
y_d_expr = Expression('exp(-20*(pow(x[0] - 0.5 - 0.25*cos(2*pi*t), 2) + pow(x[1] - 0.5 - 0.25*sin(2*pi*t), 2)))', degree=1, t=0.0)


for i in range(len(t_array)):
	t = t_array[i]
	y_d_expr.t = t

	y = states[i]
	if i == 0:
		y_prev = Function(V)
	else:
		y_prev = states[i-1]
	p = adjoints[i]
	u = controls[i]

	state_eq = Constant(1/dt)*(y - y_prev)*p*dx + inner(grad(y), grad(p))*dx - u*p*dx

	e.append(state_eq)
	y_d.append(interpolate(y_d_expr, V))

	J_summands.append(Constant(0.5*dt)*(y - y_d[i])*(y - y_d[i])*dx + Constant(0.5*lambd)*u*u*dx)

J = summation(J_summands)

optimization_problem = OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)
optimization_problem.solve()


end_time = time.time()
print('Ellapsed time ' + str(end_time - start_time) + ' s')

u_file = File('./pvd/u.pvd')
y_file = File('./pvd/y.pvd')
temp_u = Function(V)
temp_y = Function(V)

for i in range(len(t_array)):
	t = t_array[i]

	temp_u.vector()[:] = controls[i].vector()[:]
	u_file << temp_u, t

	temp_y.vector()[:] = states[i].vector()[:]
	y_file << temp_y, t
