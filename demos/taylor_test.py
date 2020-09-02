# Copyright (C) 2020 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""
Created on 25/08/2020, 14.36

@author: blauths
"""

from fenics import *

import cashocs



config = cashocs.create_config('./config.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
p = Function(V)
u = Function(V)
u_orig = Function(V)
u_orig.vector()[:] = u.vector()[:]

F = inner(grad(y), grad(p))*dx - u*p*dx
bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)

cashocs.verification.taylor_tests.control_gradient_test(ocp, [u_orig])


# Ju = ocp.reduced_cost_functional.evaluate()
# dJu = ocp.compute_gradient()[0]
#
# h = Function(V)
# h.vector()[:] = np.random.rand(V.dim())
#
# dJ_h = ocp.form_handler.scalar_product([dJu], [h])
#
# epsilons = [0.01 / 2**i for i in range(4)]
# residuals = []
#
# for eps in epsilons:
# 	u.vector()[:] = u_orig.vector()[:] + eps * h.vector()[:]
# 	ocp.state_problem.has_solution = False
# 	Jv = ocp.reduced_cost_functional.evaluate()
#
# 	res = abs(Jv - Ju - eps*dJ_h)
# 	residuals.append(res)
#
#
# def convergence_rates(E_values, eps_values, show=True):
# 	from numpy import log
# 	r = []
# 	for i in range(1, len(eps_values)):
# 		r.append(log(E_values[i] / E_values[i - 1])
#                  / log(eps_values[i] / eps_values[i - 1]))
# 	if show:
# 		print("Computed convergence rates: {}".format(r))
# 	return r
#
# convergence_rates(residuals, epsilons)
