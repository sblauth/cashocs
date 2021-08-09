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

"""Tests for shape optimization problems.

"""

import os
import subprocess

import numpy as np
import pytest
from fenics import *

import cashocs


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

    return Constant(0.5) * (grad(u) + grad(u).T)


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

    return grad(u) - outer(grad(u) * n, n)


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

    return div(u) - inner(grad(u) * n, n)


rng = np.random.RandomState(300696)
dir_path = os.path.dirname(os.path.realpath(__file__))
# config = cashocs.load_config(dir_path + "/config_sop.ini")

meshlevel = 10
degree = 1
dim = 2
mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
initial_coordinates = mesh.coordinates().copy()
dx = Measure("dx", mesh)
ds = Measure("ds", mesh)

boundary = CompiledSubDomain("on_boundary")
boundaries = MeshFunction("size_t", mesh, dim=1)
boundary.mark(boundaries, 1)

V = FunctionSpace(mesh, "CG", 1)

bcs = DirichletBC(V, Constant(0), boundaries, 1)

x = SpatialCoordinate(mesh)
f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1
# f = Expression('2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1', degree=4, domain=mesh)

u = Function(V)
p = Function(V)

e = inner(grad(u), grad(p)) * dx - f * p * dx

J = u * dx


def test_move_mesh():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    V = VectorFunctionSpace(mesh, "CG", 1)
    offset = rng.rand(2)
    trafo = interpolate(Constant(offset), V)
    sop.mesh_handler.move_mesh(trafo)

    deformed_coordinates = np.zeros(initial_coordinates.shape)
    deformed_coordinates[:, 0] = initial_coordinates[:, 0] + offset[0]
    deformed_coordinates[:, 1] = initial_coordinates[:, 1] + offset[1]
    assert np.alltrue(abs(mesh.coordinates()[:, :] - deformed_coordinates) < 1e-15)

    sop.mesh_handler.revert_transformation()
    assert np.alltrue(abs(mesh.coordinates()[:, :] - initial_coordinates) < 1e-15)

    trafo.vector()[:] = rng.uniform(-1e3, 1e3, V.dim())
    sop.mesh_handler.move_mesh(trafo)
    assert np.alltrue(abs(mesh.coordinates()[:, :] - initial_coordinates) < 1e-15)


def test_shape_derivative_unconstrained():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    n = FacetNormal(mesh)

    CG1 = VectorFunctionSpace(mesh, "CG", 1)
    defo = TestFunction(CG1)

    J1 = Constant(1) * dx
    J2 = Constant(1) * ds

    sop1 = cashocs.ShapeOptimizationProblem(e, bcs, J1, u, p, boundaries, config)
    sop1.state_problem.has_solution = True
    sop1.adjoint_problem.has_solution = True
    cashocs_sd_1 = assemble(sop1.form_handler.shape_derivative)[:]
    exact_sd_1 = assemble(div(defo) * dx)[:]

    sop2 = cashocs.ShapeOptimizationProblem(e, bcs, J2, u, p, boundaries, config)
    sop2.state_problem.has_solution = True
    sop2.adjoint_problem.has_solution = True
    cashocs_sd_2 = assemble(sop2.form_handler.shape_derivative)[:]
    exact_sd_2 = assemble(t_div(defo, n) * ds)[:]

    assert np.allclose(cashocs_sd_1, exact_sd_1)
    assert np.allclose(cashocs_sd_2, exact_sd_2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_shape_derivative_constrained():
    """Note, that the warning raised by cashocs is also dealt with in this test.
    No need to show a warning in pytest.

    """
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    n = FacetNormal(mesh)

    CG1 = VectorFunctionSpace(mesh, "CG", 1)
    defo = TestFunction(CG1)

    x = SpatialCoordinate(mesh)
    u_d_coord = pow(x[0], 2) + pow(x[1], 4) - pi
    u_d_expr = Expression("pow(x[0], 2) + pow(x[1], 4) - pi", degree=4, domain=mesh)
    u_d_func = interpolate(u_d_expr, V)

    J_coord = (u - u_d_coord) * (u - u_d_coord) * dx
    J_expr = (u - u_d_expr) * (u - u_d_expr) * dx
    J_func = (u - u_d_func) * (u - u_d_func) * dx

    exact_shape_derivative = (
        (u - u_d_coord) * (u - u_d_coord) * div(defo) * dx
        - Constant(2) * (u - u_d_coord) * dot(grad(u_d_coord), defo) * dx
        + div(defo) * inner(grad(u), grad(p)) * dx
        - Constant(2) * inner(eps(defo) * grad(u), grad(p)) * dx
        - div(defo) * f * p * dx
        - inner(grad(f), defo) * p * dx
    )

    sop_coord = cashocs.ShapeOptimizationProblem(
        e, bcs, J_coord, u, p, boundaries, config
    )
    sop_coord.compute_adjoint_variables()
    cashocs_sd_coord = assemble(sop_coord.form_handler.shape_derivative)[:]

    config.set("ShapeGradient", "degree_estimation", "True")
    sop_expr = cashocs.ShapeOptimizationProblem(
        e, bcs, J_expr, u, p, boundaries, config
    )
    sop_expr.compute_adjoint_variables()
    cashocs_sd_expr = assemble(
        sop_expr.form_handler.shape_derivative,
        form_compiler_parameters={"quadrature_degree": 10},
    )[:]
    ### degree estimation is only needed to avoid pytest warnings regarding numpy. This is only a fenics problem.

    exact_sd = assemble(exact_shape_derivative)[:]
    assert np.allclose(exact_sd, cashocs_sd_coord)
    assert np.allclose(exact_sd, cashocs_sd_expr)

    # Need 2 objects, since interpolation of u_d into CG1 space does not yield 4th order polynomial
    exact_shape_derivative_func = (
        (u - u_d_func) * (u - u_d_func) * div(defo) * dx
        - Constant(2) * (u - u_d_func) * dot(grad(u_d_func), defo) * dx
        + div(defo) * inner(grad(u), grad(p)) * dx
        - Constant(2) * inner(eps(defo) * grad(u), grad(p)) * dx
        - div(defo) * f * p * dx
        - inner(grad(f), defo) * p * dx
    )

    sop_func = cashocs.ShapeOptimizationProblem(
        e, bcs, J_func, u, p, boundaries, config
    )
    sop_func.compute_adjoint_variables()
    cashocs_sd_func = assemble(sop_func.form_handler.shape_derivative)[:]

    exact_sd_func = assemble(exact_shape_derivative_func)[:]
    assert np.allclose(exact_sd_func, cashocs_sd_func)


def test_shape_gradient():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9


def test_shape_gd():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("gd", rtol=1e-2, atol=0.0, max_iter=32)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_cg_fr():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    config.set("AlgoCG", "cg_method", "FR")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("cg", rtol=1e-2, atol=0.0, max_iter=15)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_cg_pr():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    config.set("AlgoCG", "cg_method", "PR")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("cg", rtol=1e-2, atol=0.0, max_iter=25)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_cg_hs():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    config.set("AlgoCG", "cg_method", "HS")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("cg", rtol=1e-2, atol=0.0, max_iter=23)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_cg_dy():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    config.set("AlgoCG", "cg_method", "DY")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("cg", rtol=1e-2, atol=0.0, max_iter=17)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_cg_hz():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    config.set("AlgoCG", "cg_method", "HZ")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("cg", rtol=1e-2, atol=0.0, max_iter=22)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_lbfgs():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("lbfgs", rtol=1e-2, atol=0.0, max_iter=8)
    assert sop.solver.relative_norm < sop.solver.rtol


def test_shape_volume_regularization():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    config.set("Regularization", "factor_volume", "1.0")
    radius = rng.uniform(0.33, 0.66)
    config.set("Regularization", "target_volume", str(np.pi * radius ** 2))
    config.set("MeshQuality", "volume_change", "10")
    J_vol = Constant(0) * dx
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J_vol, u, p, boundaries, config)

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9

    sop.solve("lbfgs", rtol=1e-6, max_iter=50)
    max_coordinate = np.max(mesh.coordinates())
    min_coordinate = -np.min(mesh.coordinates())
    assert abs(max_coordinate - radius) < 5e-3
    assert abs(min_coordinate - radius) < 5e-3
    assert 0.5 * pow(assemble(1 * dx) - np.pi * radius ** 2, 2) < 1e-10


def test_shape_surface_regularization():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    config.set("Regularization", "factor_surface", "1.0")
    radius = rng.uniform(0.33, 0.66)
    config.set("Regularization", "target_surface", str(2 * np.pi * radius))
    config.set("MeshQuality", "volume_change", "10")
    J_vol = Constant(0) * dx
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J_vol, u, p, boundaries, config)

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9

    sop.solve("lbfgs", rtol=1e-6, max_iter=50)
    max_coordinate = np.max(mesh.coordinates())
    min_coordinate = -np.min(mesh.coordinates())
    assert abs(max_coordinate - radius) < 5e-3
    assert abs(min_coordinate - radius) < 5e-3
    assert 0.5 * pow(assemble(1 * ds) - 2 * np.pi * radius, 2) < 1e-10


def test_shape_barycenter_regularization():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    config.set("Regularization", "factor_volume", "1e2")
    config.set("Regularization", "use_initial_volume", "True")
    config.set("Regularization", "factor_barycenter", "1.0")
    pos_x = rng.uniform(0.2, 0.4)
    pos_y = rng.uniform(-0.4, -0.2)
    config.set("Regularization", "target_barycenter", str([pos_x, pos_y]))
    config.set("MeshQuality", "volume_change", "10")
    initial_volume = assemble(1 * dx)
    J_vol = Constant(0) * dx
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J_vol, u, p, boundaries, config)

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9

    sop.solve("lbfgs", rtol=1e-5, max_iter=50)

    x = SpatialCoordinate(mesh)
    volume = assemble(1 * dx)
    bc_x = assemble(x[0] * dx) / volume
    bc_y = assemble(x[1] * dx) / volume

    assert abs(volume - initial_volume) < 1e-2
    assert abs(bc_x - pos_x) < 1e-4
    assert abs(bc_y - pos_y) < 1e-4


def test_shape_barycenter_regularization_hole():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/barycenter_hole/mesh.xdmf"
    )
    config = cashocs.load_config(dir_path + "/config_sop.ini")
    V = FunctionSpace(mesh, "CG", 1)
    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
    x = SpatialCoordinate(mesh)
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    u = Function(V)
    p = Function(V)

    e = inner(grad(u), grad(p)) * dx - f * p * dx

    config.set("Regularization", "factor_volume", "1e2")
    config.set("Regularization", "use_initial_volume", "True")
    config.set("Regularization", "factor_barycenter", "1.0")
    config.set("Regularization", "measure_hole", "True")
    pos_x = rng.uniform(0.2, 0.4)
    pos_y = rng.uniform(-0.4, -0.2)
    config.set("Regularization", "target_barycenter", str([pos_x, pos_y]))
    config.set("MeshQuality", "volume_change", "10")
    J_vol = Constant(0) * dx
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J_vol, u, p, boundaries, config)

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9


def test_custom_supply_shape():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    user_sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    vfield = user_sop.get_vector_field()
    I = Identity(2)

    adjoint_form = inner(grad(p), grad(TestFunction(V))) * dx - TestFunction(V) * dx
    dJ = (
        u * div(vfield) * dx
        - inner((div(vfield) * I - 2 * eps(vfield)) * grad(u), grad(p)) * dx
        + div(f * vfield) * p * dx
    )

    user_sop.supply_custom_forms(dJ, adjoint_form, bcs)

    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9

    user_sop.supply_custom_forms(dJ, [adjoint_form], [bcs])

    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9


def test_supply_from_custom_fspace():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    user_sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    VCG = VectorFunctionSpace(mesh, "CG", 1)
    vfield = TestFunction(VCG)
    I = Identity(2)

    adjoint_form = inner(grad(p), grad(TestFunction(V))) * dx - TestFunction(V) * dx
    dJ = (
        u * div(vfield) * dx
        - inner((div(vfield) * I - 2 * eps(vfield)) * grad(u), grad(p)) * dx
        + div(f * vfield) * p * dx
    )

    user_sop.supply_custom_forms(dJ, adjoint_form, bcs)

    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9


def test_custom_shape_scalar_product():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    config.set("ShapeGradient", "damping_factor", "0.0")

    space = VectorFunctionSpace(mesh, "CG", 1)
    shape_scalar_product = (
        Constant(1)
        * inner((grad(TrialFunction(space))), (grad(TestFunction(space))))
        * dx
        + inner(TrialFunction(space), TestFunction(space)) * dx
    )

    config.set("ShapeGradient", "damping_factor", "0.2")

    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J, u, p, boundaries, config, shape_scalar_product=shape_scalar_product
    )
    sop.solve("lbfgs", rtol=1e-2, atol=0.0, max_iter=8)

    assert sop.solver.relative_norm < sop.solver.rtol

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    user_sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    vfield = user_sop.get_vector_field()
    I = Identity(2)

    adjoint_form = inner(grad(p), grad(TestFunction(V))) * dx - TestFunction(V) * dx
    dJ = (
        u * div(vfield) * dx
        - inner((div(vfield) * I - 2 * eps(vfield)) * grad(u), grad(p)) * dx
        + div(f * vfield) * p * dx
    )

    user_sop.supply_custom_forms(dJ, adjoint_form, bcs)

    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9

    user_sop.supply_custom_forms(dJ, [adjoint_form], [bcs])

    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(user_sop, rng=rng) > 1.9


def test_scaling_shape():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    J1 = u * dx
    J2 = u * u * dx
    J_list = [J1, J2]

    desired_weights = rng.rand(2).tolist()
    diff = desired_weights[1] - desired_weights[0]

    test_sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J_list, u, p, boundaries, config, desired_weights=desired_weights
    )
    val = test_sop.reduced_cost_functional.evaluate()

    assert abs(val - diff) < 1e-14

    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9


def test_scaling_shape_regularization():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    no_iterations = 5
    test_weights = rng.rand(no_iterations, 4)
    config.set("Regularization", "use_relative_scaling", "True")
    config.set("Regularization", "target_barycenter", "[1.0, 1.0, 0.0]")

    for iteration in range(no_iterations):

        mesh.coordinates()[:, :] = initial_coordinates
        mesh.bounding_box_tree().build(mesh)

        J = Constant(0) * dx

        config.set("Regularization", "factor_volume", str(test_weights[iteration, 0]))
        config.set("Regularization", "factor_surface", str(test_weights[iteration, 1]))
        config.set(
            "Regularization", "factor_curvature", str(test_weights[iteration, 2])
        )
        config.set(
            "Regularization", "factor_barycenter", str(test_weights[iteration, 3])
        )

        test_sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)

        summ = np.sum(test_weights[iteration, :])
        val = test_sop.reduced_cost_functional.evaluate()

        assert abs(val - summ) < 1e-15


def test_curvature_computation():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    config.set("Regularization", "factor_curvature", "1.0")

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)

    sop.form_handler.regularization.compute_curvature()

    kappa = sop.form_handler.regularization.kappa_curvature
    mean_curvature = assemble(sqrt(inner(kappa, kappa)) * ds) / assemble(1 * ds)

    assert abs(mean_curvature - 1) < 1e-3


def test_scalar_tracking_regularization():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    radius = rng.uniform(0.33, 0.66)
    tracking_goal = np.pi * radius ** 2
    config.set("MeshQuality", "volume_change", "10")
    J_vol = Constant(0) * dx
    J_tracking = {"integrand": Constant(1) * dx, "tracking_goal": tracking_goal}
    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J_vol, u, p, boundaries, config, scalar_tracking_forms=J_tracking
    )

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9

    sop.solve("lbfgs", rtol=1e-6, max_iter=50)
    max_coordinate = np.max(mesh.coordinates())
    min_coordinate = -np.min(mesh.coordinates())
    assert abs(max_coordinate - radius) < 5e-3
    assert abs(min_coordinate - radius) < 5e-3
    assert 0.5 * pow(assemble(1 * dx) - np.pi * radius ** 2, 2) < 1e-10


def test_scalar_tracking_norm():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    tracking_goal = rng.uniform(0.25, 0.75)
    norm_u = u * u * dx
    J_tracking = {"integrand": norm_u, "tracking_goal": tracking_goal}

    J_vol = Constant(0) * dx

    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J_vol, u, p, boundaries, config, scalar_tracking_forms=J_tracking
    )

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9

    sop.solve("lbfgs", rtol=1e-5, max_iter=50)
    assert 0.5 * pow(assemble(norm_u) - tracking_goal, 2) < 1e-14


def test_scalar_tracking_multiple():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    tracking_goals = [0.00541757222440158, 2.8726020865244792]
    norm_u = u * u * dx
    volume = Constant(1) * dx
    J_u = {"integrand": norm_u, "tracking_goal": tracking_goals[0]}
    J_volume = {"integrand": volume, "tracking_goal": tracking_goals[1]}
    J_tracking = [J_u, J_volume]

    J_vol = Constant(0) * dx

    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J_vol, u, p, boundaries, config, scalar_tracking_forms=J_tracking
    )

    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9

    sop.solve("lbfgs", rtol=1e-6, max_iter=50)
    assert 0.5 * pow(assemble(norm_u) - tracking_goals[0], 2) < 1e-13
    assert 0.5 * pow(assemble(volume) - tracking_goals[1], 2) < 1e-15


def test_scaling_scalar_only():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    J = Constant(0) * dx
    tracking_goals = rng.uniform(0.25, 0.75, 2)
    J_scalar1 = {"integrand": Constant(1) * dx, "tracking_goal": tracking_goals[0]}
    J_scalar2 = {"integrand": Constant(1) * ds, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_scalar1, J_scalar2]

    desired_weights = rng.rand(3).tolist()
    summ = np.sum(desired_weights[1:])

    test_sop = cashocs.ShapeOptimizationProblem(
        e,
        bcs,
        J,
        u,
        p,
        boundaries,
        config,
        desired_weights=desired_weights,
        scalar_tracking_forms=J_scalar,
    )
    val = test_sop.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9


def test_scaling_scalar_and_single_cost():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    J = u * dx
    tracking_goals = rng.uniform(0.25, 0.75, 2)
    J_scalar1 = {"integrand": Constant(1) * dx, "tracking_goal": tracking_goals[0]}
    J_scalar2 = {"integrand": Constant(1) * ds, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_scalar1, J_scalar2]

    desired_weights = rng.rand(3).tolist()
    summ = -desired_weights[0] + np.sum(desired_weights[1:])

    test_sop = cashocs.ShapeOptimizationProblem(
        e,
        bcs,
        J,
        u,
        p,
        boundaries,
        config,
        desired_weights=desired_weights,
        scalar_tracking_forms=J_scalar,
    )
    val = test_sop.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9


def test_scaling_all():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    J1 = u * dx
    J2 = u * u * dx
    J_list = [J1, J2]
    tracking_goals = rng.uniform(0.25, 0.75, 2)
    J_scalar1 = {"integrand": Constant(1) * dx, "tracking_goal": tracking_goals[0]}
    J_scalar2 = {"integrand": Constant(1) * ds, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_scalar1, J_scalar2]

    desired_weights = rng.rand(4).tolist()
    summ = -desired_weights[0] + np.sum(desired_weights[1:])

    test_sop = cashocs.ShapeOptimizationProblem(
        e,
        bcs,
        J_list,
        u,
        p,
        boundaries,
        config,
        desired_weights=desired_weights,
        scalar_tracking_forms=J_scalar,
    )
    val = test_sop.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(test_sop, rng=rng) > 1.9


def test_inhomogeneous_mu():
    config = cashocs.load_config(dir_path + "/config_sop.ini")

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    V = FunctionSpace(mesh, "CG", 1)

    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

    x = SpatialCoordinate(mesh)
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    u = Function(V)
    p = Function(V)

    e = inner(grad(u), grad(p)) * dx - f * p * dx

    J = u * dx
    config.set("ShapeGradient", "shape_bdry_def", "[1,2]")
    config.set("ShapeGradient", "shape_bdry_fix", "[3,4]")
    config.set("ShapeGradient", "mu_fix", "1.0")
    config.set("ShapeGradient", "mu_def", "10.0")
    config.set("ShapeGradient", "inhomogeneous", "True")
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9


def test_save_pvd_files():
    config = cashocs.load_config(dir_path + "/config_sop.ini")
    config.set("Output", "save_pvd", "True")
    config.set("Output", "save_results", "True")
    config.set("Output", "save_txt", "True")
    config.set("Output", "save_pvd_adjoint", "True")
    config.set("Output", "save_pvd_gradient", "True")
    config.set("Output", "result_dir", dir_path + "/out")
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)
    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve("lbfgs", rtol=1e-2, atol=0.0, max_iter=8)
    assert os.path.isdir(dir_path + "/out")
    assert os.path.isdir(dir_path + "/out/pvd")
    assert os.path.isfile(dir_path + "/out/history.txt")
    assert os.path.isfile(dir_path + "/out/history.json")
    assert os.path.isfile(dir_path + "/out/pvd/state_0.pvd")
    assert os.path.isfile(dir_path + "/out/pvd/state_0000007.vtu")
    assert os.path.isfile(dir_path + "/out/pvd/adjoint_0.pvd")
    assert os.path.isfile(dir_path + "/out/pvd/adjoint_0000007.vtu")
    assert os.path.isfile(dir_path + "/out/pvd/shape_gradient.pvd")
    assert os.path.isfile(dir_path + "/out/pvd/shape_gradient000007.vtu")

    subprocess.run(["rm", "-r", f"{dir_path}/out"], check=True)
