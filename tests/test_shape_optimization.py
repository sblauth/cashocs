"""
Created on 25/08/2020, 16.33

@author: blauths
"""

from fenics import *
import cashocs
import numpy as np



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


# sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
# sop.solve('gd', rtol=1e-2, atol=0.0, max_iter=35)
