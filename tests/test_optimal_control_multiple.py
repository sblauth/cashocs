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
Created on 02/09/2020, 09.13

@author: blauths
"""

from fenics import *

import cashocs



config = cashocs.create_config('./config_ocp.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
z = Function(V)
p = Function(V)
q = Function(V)
u = Function(V)
v = Function(V)

states = [y, z]
adjoints = [p, q]
controls = [u, v]

e_y = inner(grad(y), grad(p)) * dx - u * p * dx
e_z = inner(grad(z), grad(q)) * dx - (y + v) * q * dx

e = [e_y, e_z]

bcs1 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs2 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

bcs_list = [bcs1, bcs2]

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
alpha = 1e-6
beta = 1e-4
J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5) * (z - z_d) * (z - z_d) * dx \
	+ Constant(0.5*alpha) * u * u * dx + Constant(0.5*beta) * v * v * dx

ocp = cashocs.OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)



def test_control_gradient_multiple():
	assert cashocs.verification.control_gradient_test(ocp) > 1.9
	assert cashocs.verification.control_gradient_test(ocp) > 1.9
	assert cashocs.verification.control_gradient_test(ocp) > 1.9



def test_control_gd_multiple():
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('gd', rtol=1e-2, atol=0.0, max_iter=47)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_fr_multiple():
	config.set('OptimizationRoutine', 'cg_method', 'FR')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=21)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_pr_multiple():
	config.set('OptimizationRoutine', 'cg_method', 'PR')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=36)
	assert ocp.solver.relative_norm <= ocp.solver.rtol
	


def test_control_cg_hs_multiple():
	config.set('OptimizationRoutine', 'cg_method', 'HS')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=30)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_dy_multiple():
	config.set('OptimizationRoutine', 'cg_method', 'DY')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=13)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_hz_multiple():
	config.set('OptimizationRoutine', 'cg_method', 'HZ')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=26)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_bfgs_multiple():
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('bfgs', rtol=1e-2, atol=0.0, max_iter=11)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_newton_cg_multiple():
	config.set('OptimizationRoutine', 'inner_newton', 'cg')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('newton', rtol=1e-2, atol=0.0, max_iter=2)
	assert ocp.solver.relative_norm <= 1e-4



def test_control_newton_cr_multiple():
	config.set('OptimizationRoutine', 'inner_newton', 'cr')
	u.vector()[:] = 0.0
	v.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('newton', rtol=1e-2, atol=0.0, max_iter=2)
	assert ocp.solver.relative_norm <= 1e-4
