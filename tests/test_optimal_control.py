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
Created on 25/08/2020, 15.55

@author: blauths
"""

import numpy as np
from fenics import *

import cashocs



config = cashocs.create_config('./config_ocp.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p))*dx - u*p*dx
bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1, domain=mesh)
alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)

cc = [0, 100]

ocp_cc = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config, control_constraints=cc)



def test_control_gradient():
	assert cashocs.verification.control_gradient_test(ocp) > 1.9
	assert cashocs.verification.control_gradient_test(ocp) > 1.9
	assert cashocs.verification.control_gradient_test(ocp) > 1.9



def test_control_gd():
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('gd', rtol=1e-2, atol=0.0, max_iter=46)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_fr():
	config.set('AlgoCG', 'cg_method', 'FR')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=21)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_pr():
	config.set('AlgoCG', 'cg_method', 'PR')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=26)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_hs():
	config.set('AlgoCG', 'cg_method', 'HS')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=28)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_dy():
	config.set('AlgoCG', 'cg_method', 'DY')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=10)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_cg_hz():
	config.set('AlgoCG', 'cg_method', 'HZ')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('cg', rtol=1e-2, atol=0.0, max_iter=28)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_bfgs():
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('bfgs', rtol=1e-2, atol=0.0, max_iter=7)
	assert ocp.solver.relative_norm <= ocp.solver.rtol



def test_control_newton_cg():
	config.set('AlgoTNM', 'inner_newton', 'cg')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('newton', rtol=1e-2, atol=0.0, max_iter=2)
	assert ocp.solver.relative_norm <= 1e-6



def test_control_newton_cr():
	config.set('AlgoTNM', 'inner_newton', 'cr')
	u.vector()[:] = 0.0
	ocp._erase_pde_memory()
	ocp.solve('newton', rtol=1e-2, atol=0.0, max_iter=2)
	assert ocp.solver.relative_norm <= 1e-6



def test_control_gd_cc():
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('gd', rtol=1e-2, atol=0.0, max_iter=22)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_cg_fr_cc():
	config.set('AlgoCG', 'cg_method', 'FR')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('cg', rtol=1e-2, atol=0.0, max_iter=48)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_cg_pr_cc():
	config.set('AlgoCG', 'cg_method', 'PR')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('cg', rtol=1e-2, atol=0.0, max_iter=25)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_cg_hs_cc():
	config.set('AlgoCG', 'cg_method', 'HS')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('cg', rtol=1e-2, atol=0.0, max_iter=30)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_cg_dy_cc():
	config.set('AlgoCG', 'cg_method', 'DY')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('cg', rtol=1e-2, atol=0.0, max_iter=9)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_cg_hz_cc():
	config.set('AlgoCG', 'cg_method', 'HZ')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('cg', rtol=1e-2, atol=0.0, max_iter=37)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_lbfgs_cc():
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('lbfgs', rtol=1e-2, atol=0.0, max_iter=11)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_newton_cg_cc():
	config.set('AlgoTNM', 'inner_newton', 'cg')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('newton', rtol=1e-2, atol=0.0, max_iter=8)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_newton_cr_cc():
	config.set('AlgoTNM', 'inner_newton', 'cr')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('newton', rtol=1e-2, atol=0.0, max_iter=9)
	assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_pdas_gd_cc():
	config.set('AlgoPDAS', 'inner_pdas', 'gd')
	config.set('OptimizationRoutine', 'soft_exit', 'True')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('pdas', rtol=1e-2, atol=0.0, max_iter=9)

	config.set('OptimizationRoutine', 'soft_exit', 'False')

	assert ocp_cc.solver.converged
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_pdas_cg_fr_cc():
	config.set('AlgoPDAS', 'inner_pdas', 'cg')
	config.set('AlgoCG', 'cg_method', 'FR')
	config.set('OptimizationRoutine', 'soft_exit', 'True')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('pdas', rtol=1e-2, atol=0.0, max_iter=10)

	config.set('OptimizationRoutine', 'soft_exit', 'False')

	assert ocp_cc.solver.converged
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_pdas_cg_pr_cc():
	config.set('AlgoPDAS', 'inner_pdas', 'cg')
	config.set('AlgoCG', 'cg_method', 'PR')
	config.set('OptimizationRoutine', 'soft_exit', 'True')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('pdas', rtol=1e-2, atol=0.0, max_iter=11)

	config.set('OptimizationRoutine', 'soft_exit', 'False')

	assert ocp_cc.solver.converged
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_pdas_cg_dy_cc():
	config.set('AlgoPDAS', 'inner_pdas', 'cg')
	config.set('AlgoCG', 'cg_method', 'DY')
	config.set('OptimizationRoutine', 'soft_exit', 'True')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('pdas', rtol=1e-2, atol=0.0, max_iter=10)

	config.set('OptimizationRoutine', 'soft_exit', 'False')

	assert ocp_cc.solver.converged
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])



def test_control_pdas_bfgs_cc():
	config.set('AlgoPDAS', 'inner_pdas', 'lbfgs')
	config.set('OptimizationRoutine', 'soft_exit', 'True')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('pdas', rtol=1e-2, atol=0.0, max_iter=17)

	config.set('OptimizationRoutine', 'soft_exit', 'False')

	assert ocp_cc.solver.converged
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_pdas_newton():
	config.set('AlgoPDAS', 'inner_pdas', 'newton')
	config.set('OptimizationRoutine', 'soft_exit', 'True')
	u.vector()[:] = 0.0
	ocp_cc._erase_pde_memory()
	ocp_cc.solve('pdas', rtol=1e-2, atol=0.0, max_iter=10)

	config.set('OptimizationRoutine', 'soft_exit', 'False')

	assert ocp_cc.solver.converged
	assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
	assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])
