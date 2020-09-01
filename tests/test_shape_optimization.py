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
Created on 25/08/2020, 16.33

@author: blauths
"""

import pytest
from fenics import *
import cashocs
import numpy as np



def eps(u):
	"""Computes the symmetric gradient of u

	Parameters
	----------
	u : dolfin.function.function.Function

	Returns
	-------
	ufl.core.expr.Expr
		the symmetric gradient of u
	"""

	return Constant(0.5)*(grad(u) + grad(u).T)


def t_grad(u, n):
	"""Computes the tangential gradient of u

	Parameters
	----------
	u : dolfin.function.function.Function
		the argument
	n : ufl.geometry.FacetNormal
		the unit outer normal vector

	Returns
	-------
	ufl.core.expr.Expr
		the tangential gradient of u
	"""

	return grad(u) - outer(grad(u)*n, n)


def t_div(u, n):
	"""Computes the tangential divergence

	Parameters
	----------
	u : dolfin.function.function.Function
		the argument
	n : ufl.geometry.FacetNormal
		the outer unit normal vector

	Returns
	-------
	ufl.core.expr.Expr
		the tangential divergence of u
	"""

	return div(u) - inner(grad(u)*n, n)

config = cashocs.create_config('./config_sop.ini')

meshlevel = 10
degree = 1
dim = 2
mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
initial_coordinates = mesh.coordinates().copy()
dx = Measure('dx', mesh)
ds = Measure('ds', mesh)

boundary = CompiledSubDomain('on_boundary')
boundaries = MeshFunction('size_t', mesh, dim=1)
boundary.mark(boundaries, 1)

V = FunctionSpace(mesh, 'CG', 1)

bcs = DirichletBC(V, Constant(0), boundaries, 1)

x = SpatialCoordinate(mesh)
f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1
# f = Expression('2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1', degree=4, domain=mesh)

u = Function(V)
p = Function(V)

e = inner(grad(u), grad(p))*dx - f*p*dx

J = u*dx



def test_move_mesh():
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
	V = VectorFunctionSpace(mesh, 'CG', 1)
	offset = np.random.rand(2)
	trafo = interpolate(Constant(offset), V)
	sop.mesh_handler.move_mesh(trafo)
	
	deformed_coordinates = np.zeros(initial_coordinates.shape)
	deformed_coordinates[:, 0] = initial_coordinates[:, 0] + offset[0]
	deformed_coordinates[:, 1] = initial_coordinates[:, 1] + offset[1]
	assert np.alltrue(abs(mesh.coordinates()[:, :] -  deformed_coordinates ) < 1e-15)
	
	sop.mesh_handler.revert_transformation()
	assert np.alltrue(abs(mesh.coordinates()[:, :] - initial_coordinates) < 1e-15)
	
	trafo.vector()[:] = np.random.uniform(-1e3, 1e3, V.dim())
	sop.mesh_handler.move_mesh(trafo)
	assert np.alltrue(abs(mesh.coordinates()[:, :] - initial_coordinates) < 1e-15)


def test_shape_derivative_unconstrained():
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	n = FacetNormal(mesh)
	
	CG1 = VectorFunctionSpace(mesh, 'CG', 1)
	defo = TestFunction(CG1)
	
	J1 = Constant(1)*dx
	J2 = Constant(1)*ds
	
	sop1 = cashocs.ShapeOptimizationProblem(e, bcs, J1, u, p, boundaries, config)
	sop1.state_problem.has_solution = True
	sop1.adjoint_problem.has_solution = True
	cashocs_sd_1 = assemble(sop1.shape_form_handler.shape_derivative)[:]
	exact_sd_1 = assemble(div(defo)*dx)[:]
	
	sop2 = cashocs.ShapeOptimizationProblem(e, bcs, J2, u, p, boundaries, config)
	sop2.state_problem.has_solution = True
	sop2.adjoint_problem.has_solution = True
	cashocs_sd_2 = assemble(sop2.shape_form_handler.shape_derivative)[:]
	exact_sd_2 = assemble(t_div(defo, n)*ds)[:]
	
	assert np.allclose(cashocs_sd_1, exact_sd_1)
	assert np.allclose(cashocs_sd_2, exact_sd_2)



@pytest.mark.filterwarnings('ignore::UserWarning')
def test_shape_derivative_constrained():
	"""Note, that the warning raised by cashocs is also dealt with in this test.
	No need to show a warning in pytest.
	
	"""
	
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	n = FacetNormal(mesh)
	
	CG1 = VectorFunctionSpace(mesh, 'CG', 1)
	defo = TestFunction(CG1)
	
	x = SpatialCoordinate(mesh)
	u_d_coord = pow(x[0], 2) + pow(x[1], 4) - pi
	u_d_expr = Expression('pow(x[0], 2) + pow(x[1], 4) - pi', degree=4, domain=mesh)
	u_d_func = interpolate(u_d_expr, V)
	
	J_coord = (u - u_d_coord)*(u - u_d_coord)*dx
	J_expr = (u - u_d_expr)*(u - u_d_expr)*dx
	J_func = (u - u_d_func)*(u - u_d_func)*dx
	
	exact_shape_derivative = (u - u_d_coord)*(u - u_d_coord)*div(defo)*dx - Constant(2)*(u - u_d_coord)*dot(grad(u_d_coord), defo)*dx \
							 + div(defo)*inner(grad(u), grad(p))*dx - Constant(2)*inner(eps(defo)*grad(u), grad(p))*dx - div(defo)*f*p*dx \
							 - inner(grad(f), defo)*p*dx
	
	sop_coord = cashocs.ShapeOptimizationProblem(e, bcs, J_coord, u, p, boundaries, config)
	sop_coord.compute_adjoint_variables()
	cashocs_sd_coord = assemble(sop_coord.shape_form_handler.shape_derivative)[:]
	
	config.set('ShapeGradient', 'degree_estimation', 'True')
	sop_expr = cashocs.ShapeOptimizationProblem(e, bcs, J_expr, u, p, boundaries, config)
	sop_expr.compute_adjoint_variables()
	cashocs_sd_expr = assemble(sop_expr.shape_form_handler.shape_derivative, form_compiler_parameters={'quadrature_degree' : 10})[:]
	config.set('ShapeGradient', 'degree_estimation', 'False')
	
	exact_sd = assemble(exact_shape_derivative)[:]
	assert np.allclose(exact_sd, cashocs_sd_coord)
	assert np.allclose(exact_sd, cashocs_sd_expr)
	
	# Need 2 objects, since interpolation of u_d into CG1 space does not yield 4th order polynomial
	exact_shape_derivative_func = (u - u_d_func)*(u - u_d_func)*div(defo)*dx - Constant(2)*(u - u_d_func)*dot(grad(u_d_func), defo)*dx \
							 + div(defo)*inner(grad(u), grad(p))*dx - Constant(2)*inner(eps(defo)*grad(u), grad(p))*dx - div(defo)*f*p*dx \
							 - inner(grad(f), defo)*p*dx
	
	sop_func = cashocs.ShapeOptimizationProblem(e, bcs, J_func, u, p, boundaries, config)
	sop_func.compute_adjoint_variables()
	cashocs_sd_func = assemble(sop_func.shape_form_handler.shape_derivative)[:]
	
	exact_sd_func = assemble(exact_shape_derivative_func)[:]
	assert np.allclose(exact_sd_func, cashocs_sd_func)



def test_shape_gradient():
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
	assert cashocs.verification.shape_gradient_test(sop) > 1.9
	assert cashocs.verification.shape_gradient_test(sop) > 1.9
	assert cashocs.verification.shape_gradient_test(sop) > 1.9



def test_shape_gd():
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
	sop.solve('gd', rtol=1e-2, atol=0.0, max_iter=35)
	assert sop.solver.relative_norm < sop.solver.rtol



def test_shape_cg():
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
	sop.solve('cg', rtol=1e-2, atol=0.0, max_iter=20)
	assert sop.solver.relative_norm < sop.solver.rtol



def test_shape_lbfgs():
	mesh.coordinates()[:, :] = initial_coordinates
	mesh.bounding_box_tree().build(mesh)
	sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
	sop.solve('lbfgs', rtol=1e-2, atol=0.0, max_iter=10)
	assert sop.solver.relative_norm < sop.solver.rtol

	
