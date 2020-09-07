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
Created on 02/09/2020, 07.41

@author: blauths
"""

import fenics
import numpy as np

import cashocs



mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
V = fenics.FunctionSpace(mesh, 'CG', 1)



def test_summation():
	a = [1,2,3,4]

	dim = 3
	funcs = []
	test = fenics.TestFunction(V)

	for i in range(dim):
		temp = fenics.Function(V)
		temp.vector()[:] = np.random.rand(V.dim())
		funcs.append(temp)

	F = cashocs.utils.summation([funcs[i]*test*dx for i in range(dim)])
	F_exact = (funcs[0] + funcs[1] + funcs[2])*test*dx

	b = fenics.assemble(F)[:]
	b_exact = fenics.assemble(F_exact)[:]

	assert cashocs.utils.summation(a) == 10
	assert abs(cashocs.utils.summation(b) - np.sum(b)) < 1e-14
	assert np.allclose(b, b_exact)



def test_multiplication():
	a = [1,2,3,4]

	dim = 3
	funcs = []
	test = fenics.TestFunction(V)

	for i in range(dim):
		temp = fenics.Function(V)
		temp.vector()[:] = np.random.rand(V.dim())
		funcs.append(temp)

	F = cashocs.utils.multiplication([funcs[i] for i in range(dim)])*test*dx
	F_exact = (funcs[0]*funcs[1]*funcs[2])*test*dx

	b = fenics.assemble(F)[:]
	b_exact = fenics.assemble(F_exact)[:]

	assert cashocs.utils.multiplication(a) == 24
	assert abs(cashocs.utils.multiplication(b) - np.prod(b)) < 1e-14
	assert np.allclose(b, b_exact)



def test_empty_measure():
	trial = fenics.TrialFunction(V)
	test = fenics.TestFunction(V)
	fun = fenics.Function(V)
	fun.vector()[:] = np.random.rand(V.dim())

	d1 = cashocs.utils.EmptyMeasure(dx)
	d2 = cashocs.utils.EmptyMeasure(ds)

	a = trial*test*d1 + trial*test*d2
	L = fun*test*d1 + fun*test*d2
	F = fun*d1 + fun*d2

	A = fenics.assemble(a)
	b = fenics.assemble(L)
	c = fenics.assemble(F)

	assert np.max(np.abs(A.array())) == 0.0
	assert np.max(np.abs(b[:])) == 0.0
	assert c == 0.0



def test_create_measure():
	meas = cashocs.utils.generate_measure([1,2,3], ds)
	test = ds(1) + ds(2) + ds(3)

	assert abs(fenics.assemble(1*meas) - 3) < 1e-14
	for i in range(3):
		assert meas._measures[i] == test._measures[i]



def test_create_config():
	config = cashocs.create_config('./test_config.ini')

	assert config.getint('A', 'a') == 1
	assert config.getfloat('A', 'b') == 3.14
	assert config.getboolean('A', 'c') == True
	assert config.get('A', 'd') == 'algorithm'

	assert config.getint('B', 'a') == 2
	assert config.getfloat('B', 'b') == 6.28
	assert config.getboolean('B', 'c') == False
	assert config.get('B', 'd') == 'cashocs'



def test_create_bcs():
	trial = fenics.TrialFunction(V)
	test = fenics.TestFunction(V)
	a = fenics.inner(fenics.grad(trial), fenics.grad(test))*dx
	L = fenics.Constant(1)*test*dx

	bc_val = np.random.rand()
	bc1 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 1)
	bc2 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 2)
	bc3 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 3)
	bc4 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 4)
	bcs_ex = [bc1, bc2, bc3, bc4]
	bcs = cashocs.create_bcs_list(V, fenics.Constant(bc_val), boundaries, [1,2,3,4])

	u_ex = fenics.Function(V)
	u = fenics.Function(V)

	fenics.solve(a==L, u_ex, bcs_ex)
	fenics.solve(a==L, u, bcs)

	assert np.allclose(u_ex.vector()[:], u.vector()[:])
	assert abs(u(0, 0) - bc_val) < 1e-14



def test_interpolator():
	W = fenics.FunctionSpace(mesh, 'CG', 2)
	X = fenics.FunctionSpace(mesh, 'DG', 0)

	interp_W = cashocs.utils.Interpolator(V, W)
	interp_X = cashocs.utils.Interpolator(V, X)

	func_V = fenics.Function(V)
	func_V.vector()[:] = np.random.rand(V.dim())

	fen_W = fenics.interpolate(func_V, W)
	fen_X = fenics.interpolate(func_V, X)

	cas_W = interp_W.interpolate(func_V)
	cas_X = interp_X.interpolate(func_V)

	assert np.allclose(fen_W.vector()[:], cas_W.vector()[:])
	assert np.allclose(fen_X.vector()[:], cas_X.vector()[:])
