# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for optimal control problems / algorithms."""

from collections import namedtuple

from fenics import *
import numpy as np
import pytest

import cashocs
from cashocs._exceptions import InputError
from cashocs._exceptions import NotConvergedError


@pytest.fixture
def geometry():
    Geometry = namedtuple("Geometry", "mesh boundaries dx ds")
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(10)
    geom = Geometry(mesh, boundaries, dx, ds)

    return geom


@pytest.fixture
def CG1(geometry):
    return FunctionSpace(geometry.mesh, "CG", 1)


@pytest.fixture
def y(CG1):
    return Function(CG1)


@pytest.fixture
def p(CG1):
    return Function(CG1)


@pytest.fixture
def u(CG1):
    return Function(CG1)


@pytest.fixture
def F(y, u, p, geometry):
    return dot(grad(y), grad(p)) * geometry.dx - u * p * geometry.dx


@pytest.fixture
def bcs(CG1, geometry):
    return cashocs.create_dirichlet_bcs(
        CG1, Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )


@pytest.fixture
def J(y, y_d, u, geometry):
    alpha = 1e-6
    return cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
    )


@pytest.fixture
def ocp(F, bcs, J, y, u, p, config_ocp):
    return cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)


@pytest.fixture
def cc():
    return [0, 100]


@pytest.fixture
def ocp_cc(F, bcs, J, y, u, p, config_ocp, cc):
    return cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config=config_ocp, control_constraints=cc
    )


def test_control_constraints_handling(geometry, config_ocp, F, bcs, J, y, p, cc):
    cg1_elem = FiniteElement("CG", geometry.mesh.ufl_cell(), 1)
    vcg1_elem = VectorElement("CG", geometry.mesh.ufl_cell(), 1)
    vcg2_elem = VectorElement("CG", geometry.mesh.ufl_cell(), 2)
    real_elem = FiniteElement("R", geometry.mesh.ufl_cell(), 0)
    dg0_elem = FiniteElement("DG", geometry.mesh.ufl_cell(), 0)
    vdg0_elem = VectorElement("DG", geometry.mesh.ufl_cell(), 0)
    dg2_elem = FiniteElement("DG", geometry.mesh.ufl_cell(), 2)
    rt_elem = FiniteElement("RT", geometry.mesh.ufl_cell(), 1)

    mixed_elem = MixedElement([cg1_elem, dg0_elem, vdg0_elem])
    pass_elem = MixedElement([cg1_elem, real_elem, dg0_elem, vcg1_elem, mixed_elem])
    fail_elem1 = MixedElement([mixed_elem, cg1_elem, vdg0_elem, real_elem, rt_elem])
    fail_elem2 = MixedElement([dg2_elem, mixed_elem, cg1_elem, vdg0_elem, real_elem])
    fail_elem3 = MixedElement([mixed_elem, cg1_elem, vcg2_elem, vdg0_elem, real_elem])

    pass_space = FunctionSpace(geometry.mesh, pass_elem)
    pass_control = Function(pass_space)

    fail_space1 = FunctionSpace(geometry.mesh, fail_elem1)
    fail_space2 = FunctionSpace(geometry.mesh, fail_elem2)
    fail_space3 = FunctionSpace(geometry.mesh, fail_elem3)
    fail_control1 = Function(fail_space1)
    fail_control2 = Function(fail_space2)
    fail_control3 = Function(fail_space3)

    cashocs.OptimalControlProblem(
        F, bcs, J, y, pass_control, p, config=config_ocp, control_constraints=cc
    )

    with pytest.raises(InputError):
        cashocs.OptimalControlProblem(
            F, bcs, J, y, fail_control1, p, config=config_ocp, control_constraints=cc
        )
    with pytest.raises(InputError):
        cashocs.OptimalControlProblem(
            F, bcs, J, y, fail_control2, p, config=config_ocp, control_constraints=cc
        )
    with pytest.raises(InputError):
        cashocs.OptimalControlProblem(
            F, bcs, J, y, fail_control3, p, config=config_ocp, control_constraints=cc
        )


def test_control_gradient(rng, ocp, u):
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, [u], rng=rng) > 1.9


def test_control_gd(ocp):
    ocp.solve(algorithm="gd", rtol=1e-2, atol=0.0, max_iter=46)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


@pytest.fixture
def set_cg_tolerances(config_ocp):
    cg_method = config_ocp.get("AlgoCG", "cg_method")
    iterations = {"FR": "20", "PR": "25", "HS": "27", "DY": "9", "HZ": "27"}
    config_ocp.set("OptimizationRoutine", "max_iter", iterations[cg_method])


def test_conjugate_gradient_solver(setup_cg_method, set_cg_tolerances, ocp):
    print(ocp.db.config.get("AlgoCG", "cg_method"))
    print(ocp.db.config.get("OptimizationRoutine", "max_iter"))
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_restart_periodic(ocp):
    ocp.config.set("AlgoCG", "cg_method", "DY")
    ocp.config.set("AlgoCG", "cg_periodic_restart", "True")
    ocp.config.set("AlgoCG", "cg_periodic_its", "5")
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=10)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_restart_relative(ocp):
    ocp.config.set("AlgoCG", "cg_method", "DY")
    ocp.config.set("AlgoCG", "cg_relative_restart", "True")
    ocp.config.set("AlgoCG", "cg_restart_tol", "1.0")
    ocp.solve(algorithm="ncg", rtol=1e-2, atol=0.0, max_iter=24)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_bfgs(ocp):
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=7)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_bfgs_restarted(ocp):
    ocp.config.set("AlgoLBFGS", "bfgs_periodic_restart", "2")
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=17)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_newton_solver(setup_newton_method, ocp):
    ocp.solve(algorithm="newton", rtol=1e-2, atol=0.0, max_iter=2)
    assert ocp.solver.relative_norm <= 1e-6


def test_control_gd_cc(ocp_cc, cc):
    ocp_cc.solve(algorithm="gd", rtol=1e-2, atol=0.0, max_iter=22)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] >= cc[0])
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] <= cc[1])


@pytest.fixture
def set_cg_tolerances_constrained(config_ocp):
    cg_method = config_ocp.get("AlgoCG", "cg_method")
    iterations = {"FR": "47", "PR": "24", "HS": "30", "DY": "9", "HZ": "37"}
    config_ocp.set("OptimizationRoutine", "max_iter", iterations[cg_method])


def test_conjugate_gradient_solver_constrained(
    setup_cg_method, set_cg_tolerances_constrained, ocp_cc, cc
):
    ocp_cc.solve(algorithm="ncg", rtol=1e-2, atol=0.0)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] >= cc[0])
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] <= cc[1])


def test_control_lbfgs_cc(ocp_cc, cc):
    ocp_cc.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=11)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] >= cc[0])
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] <= cc[1])


def test_newton_solver_constrained(setup_newton_method, ocp_cc, cc):
    ocp_cc.solve(algorithm="newton", rtol=1e-2, atol=0.0, max_iter=9)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] >= cc[0])
    assert np.all(ocp_cc.db.function_db.controls[0].vector()[:] <= cc[1])


def test_custom_supply_control(CG1, geometry, y, u, p, y_d, rng, bcs, ocp):
    adjoint_form = (
        inner(grad(p), grad(TestFunction(CG1))) * geometry.dx
        - (y - y_d) * TestFunction(CG1) * geometry.dx
    )
    dJ = (
        Constant(1e-6) * u * TestFunction(CG1) * geometry.dx
        + TestFunction(CG1) * p * geometry.dx
    )

    ocp.supply_custom_forms(dJ, adjoint_form, bcs)

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scalar_norm_optimization(rng, config_ocp, y, geometry, F, bcs, u, p):
    config_ocp.set("OptimizationRoutine", "algorithm", "bfgs")
    config_ocp.set("OptimizationRoutine", "rtol", "1e-3")

    u.vector().vec().set(1e-3)
    u.vector().apply("")

    norm_y = y * y * geometry.dx
    tracking_goal = rng.uniform(0.25, 0.75)
    J = cashocs.ScalarTrackingFunctional(norm_y, tracking_goal)
    config_ocp.set("LineSearch", "initial_stepsize", "4e3")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    ocp.solve(algorithm="bfgs", rtol=1e-3)

    assert 0.5 * pow(assemble(norm_y) - tracking_goal, 2) < 1e-15

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scalar_tracking_weight(rng, geometry, config_ocp, F, bcs, y, u, p):
    u.vector().vec().set(1e-3)
    u.vector().apply("")

    norm_y = y * y * geometry.dx
    tracking_goal = rng.uniform(0.25, 0.75)
    weight = rng.uniform(0.1, 1e1)
    J = cashocs.ScalarTrackingFunctional(norm_y, tracking_goal, weight=1.0)
    config_ocp.set("LineSearch", "initial_stepsize", "4e3")

    test_ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    test_ocp.compute_state_variables()
    initial_function_value = 0.5 * pow(assemble(norm_y) - tracking_goal, 2)
    assert cashocs.verification.control_gradient_test(test_ocp, rng=rng) > 1.9

    J = cashocs.ScalarTrackingFunctional(
        norm_y, tracking_goal, weight=weight / initial_function_value
    )

    test_ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    test_ocp.compute_state_variables()
    val = test_ocp.reduced_cost_functional.evaluate()
    assert np.abs(val - weight) < 1e-15


def test_scalar_multiple_norms(rng, config_ocp, geometry, F, bcs, y, u, p):
    config_ocp.set("OptimizationRoutine", "algorithm", "bfgs")
    config_ocp.set("OptimizationRoutine", "rtol", "1e-6")
    config_ocp.set("OptimizationRoutine", "max_iter", "500")

    u.vector().vec().set(40.0)
    u.vector().apply("")

    norm_y = y * y * geometry.dx
    norm_u = u * u * geometry.dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = cashocs.ScalarTrackingFunctional(norm_y, tracking_goals[0])
    J_u = cashocs.ScalarTrackingFunctional(norm_u, tracking_goals[1])
    J = [J_y, J_u]
    config_ocp.set("LineSearch", "initial_stepsize", "1e-4")

    test_ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    test_ocp.solve(algorithm="bfgs", rtol=1e-6, max_iter=500)

    assert 0.5 * pow(assemble(norm_y) - tracking_goals[0], 2) < 1e-2
    assert 0.5 * pow(assemble(norm_u) - tracking_goals[1], 2) < 1e-4

    assert cashocs.verification.control_gradient_test(test_ocp, rng=rng) > 1.9


def test_different_spaces(config_ocp):
    config_ocp.set("OptimizationRoutine", "algorithm", "bfgs")

    parameters["ghost_mode"] = "shared_vertex"
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 1)

    y = Function(V)
    p = Function(W)
    u = Function(V)

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2
    flux = 1 / 2 * (inner(grad(u)("+") + grad(u)("-"), n("+")))
    alpha = 1e3
    gamma = 1e3
    e = (
        dot(grad(p), grad(y)) * dx
        - dot(avg(grad(p)), jump(y, n)) * dS
        - dot(jump(p, n), avg(grad(y))) * dS
        + Constant(alpha) / h_avg * dot(jump(p, n), jump(y, n)) * dS
        - dot(grad(p), y * n) * ds
        - dot(p * n, grad(y)) * ds
        + (Constant(gamma) / h) * p * y * ds
        - u * p * dx
    )

    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    lambd = 1e-6
    y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)

    J = cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * lambd) * u * u * dx
    )

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config_ocp)
    ocp.solve(algorithm="bfgs")
    assert ocp.solver.relative_norm <= ocp.solver.rtol

    parameters["ghost_mode"] = "none"


def test_nonlinear_state_eq(rng, CG1, geometry, y, u, p, config_ocp, bcs, J):
    initial_guess = Function(CG1)
    F = (
        dot(grad(y), grad(p)) * geometry.dx
        + pow(y, 3) * p * geometry.dx
        - u * p * geometry.dx
    )
    config_ocp.set("StateSystem", "is_linear", "False")
    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config=config_ocp, initial_guess=[initial_guess]
    )
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_riesz_scalar_products(rng, CG1, geometry, config_ocp, F, bcs, J, y, u, p):
    riesz_scalar_product = TrialFunction(CG1) * TestFunction(CG1) * geometry.dx
    ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J,
        y,
        u,
        p,
        config=config_ocp,
        riesz_scalar_products=riesz_scalar_product,
    )
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9

    ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J,
        y,
        u,
        p,
        config=config_ocp,
        riesz_scalar_products=[riesz_scalar_product],
    )
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_callbacks(ocp, u, CG1):
    def pre_function():
        u.vector().vec().set(1.0)
        u.vector().apply("")

    def post_function():
        u.vector().vec().set(-1.0)
        u.vector().apply("")

    grad = Function(CG1)
    grad.vector().vec().aypx(0.0, ocp.compute_gradient()[0].vector().vec())
    grad.vector().apply("")

    ocp.inject_pre_post_callback(pre_function, post_function)
    assert ocp.db.callback.pre_callback == pre_function
    assert ocp.db.callback.post_callback == post_function

    ocp.compute_state_variables()
    assert np.max(np.abs(u.vector()[:] - 1.0)) < 1e-15

    injected_grad = ocp.compute_gradient()
    assert np.max(np.abs(u.vector()[:] - (-1.0))) < 1e-15

    assert np.max(np.abs(grad.vector()[:] - injected_grad[0].vector()[:])) > 1e-3


def test_scaling_control(rng, F, bcs, y, u, p, y_d, config_ocp, geometry):
    u.vector().vec().set(1e-2)
    u.vector().apply("")

    J1 = cashocs.IntegralFunctional(Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx)
    J2 = cashocs.IntegralFunctional(Constant(0.5) * u * u * geometry.dx)
    J_list = [J1, J2]

    desired_weights = rng.rand(2).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F, bcs, J_list, y, u, p, config=config_ocp, desired_weights=desired_weights
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14
    assert abs(J_list[0].evaluate() - desired_weights[0]) < 1e-14
    assert abs(J_list[1].evaluate() - desired_weights[1]) < 1e-14

    assert cashocs.verification.control_gradient_test(test_ocp, rng=rng) > 1.9


def test_scaling_scalar_only(rng, F, bcs, y, u, p, config_ocp, geometry):
    u.vector().vec().set(40.0)
    u.vector().apply("")

    norm_y = y * y * geometry.dx
    norm_u = u * u * geometry.dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = cashocs.ScalarTrackingFunctional(norm_y, tracking_goals[0])
    J_u = cashocs.ScalarTrackingFunctional(norm_u, tracking_goals[1])
    J = [J_y, J_u]

    desired_weights = rng.rand(2).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J,
        y,
        u,
        p,
        config=config_ocp,
        desired_weights=desired_weights,
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.control_gradient_test(test_ocp, rng=rng) > 1.9


def test_scaling_scalar_and_single_cost(
    rng, F, bcs, y, u, p, y_d, config_ocp, geometry
):
    u.vector().vec().set(40.0)
    u.vector().apply("")

    norm_y = y * y * geometry.dx
    norm_u = u * u * geometry.dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J = cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5 * 1e-6) * u * u * geometry.dx
    )
    J_y = cashocs.ScalarTrackingFunctional(norm_y, tracking_goals[0])
    J_u = cashocs.ScalarTrackingFunctional(norm_u, tracking_goals[1])
    J_test = [J, J_y, J_u]

    desired_weights = rng.rand(3).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J_test,
        y,
        u,
        p,
        config=config_ocp,
        desired_weights=desired_weights,
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.control_gradient_test(test_ocp, rng=rng) > 1.9


def test_scaling_all(rng, F, bcs, y, u, p, y_d, config_ocp, geometry):
    u.vector().vec().set(40.0)
    u.vector().apply("")

    norm_y = y * y * geometry.dx
    norm_u = u * u * geometry.dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = cashocs.ScalarTrackingFunctional(norm_y, tracking_goals[0])
    J_u = cashocs.ScalarTrackingFunctional(norm_u, tracking_goals[1])
    J1 = cashocs.IntegralFunctional(Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx)
    J2 = cashocs.IntegralFunctional(Constant(0.5) * u * u * geometry.dx)
    J_list = [J1, J2, J_y, J_u]

    desired_weights = rng.rand(4).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J_list,
        y,
        u,
        p,
        config=config_ocp,
        desired_weights=desired_weights,
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.control_gradient_test(test_ocp, rng=rng) > 1.9


def test_iterative_gradient(rng, ocp):
    ocp.config.set("OptimizationRoutine", "gradient_method", "iterative")
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9


def test_small_stepsize1(ocp):
    ocp.config.set("LineSearch", "initial_stepsize", "1e-8")
    with pytest.raises(NotConvergedError) as e_info:
        ocp.solve(algorithm="gd", rtol=1e-2, atol=0.0, max_iter=2)
    assert "Armijo rule failed." in str(e_info.value)


def test_control_bcs(rng, geometry, CG1, F, bcs, J, y, u, p, config_ocp):
    value = rng.rand()
    control_bcs_list = cashocs.create_dirichlet_bcs(
        CG1, Constant(value), geometry.boundaries, [1, 2, 3, 4]
    )
    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config=config_ocp, control_bcs_list=control_bcs_list
    )
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=9)
    assert np.sqrt(assemble(pow(u - value, 2) * geometry.ds)) < 1e-15
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_safeguard_gd(ocp):
    ocp.config.set("LineSearch", "safeguard_stepsize", "True")
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=50)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


@pytest.mark.parametrize(
    "polynomial_model, iterations", [("cubic", 42), ("quadratic", 44)]
)
def test_polynomial_stepsize(ocp, polynomial_model, iterations):
    ocp.config.set("LineSearch", "method", "polynomial")
    ocp.config.set("LineSearch", "polynomial_model", polynomial_model)
    ocp.solve(algorithm="gd", rtol=1e-2, atol=0.0, max_iter=iterations)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_damped_bfgs(geometry, y, p, bcs, config_ocp):
    U = FunctionSpace(geometry.mesh, "CG", 1)
    u0 = Function(U)
    u1 = Function(U)

    u0.vector().vec().set(-0.5)
    u0.vector().apply("")
    u1.vector().vec().set(-0.5)
    u1.vector().apply("")

    F = y * p * geometry.dx

    J = cashocs.IntegralFunctional(
        (20 + (u0**2 - 10 * cos(2 * np.pi * u0)) + (u1**2 - 10 * cos(2 * np.pi * u1)))
        * geometry.dx
    )

    config_ocp.set("LineSearch", "initial_stepsize", "1e-2")
    config_ocp.set("OptimizationRoutine", "rtol", "1e-7")
    with pytest.raises(NotConvergedError) as e_info:
        config_ocp.set("AlgoLBFGS", "damped", "False")
        ocp = cashocs.OptimalControlProblem(
            F, bcs, J, y, [u0, u1], p, config=config_ocp
        )
        ocp.solve()
        MPI.barrier(MPI.comm_world)
        assert "Armijo rule failed." in str(e_info)

    u0.vector().vec().set(-0.5)
    u0.vector().apply("")
    u1.vector().vec().set(-0.5)
    u1.vector().apply("")
    config_ocp.set("AlgoLBFGS", "damped", "True")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, [u0, u1], p, config=config_ocp)
    ocp.solve()
    MPI.barrier(MPI.comm_world)

    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_optimal_control_with_custom_preconditioner(geometry, config_ocp):
    mesh = geometry.mesh
    dx = geometry.dx

    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, v_elem * p_elem)
    W = VectorFunctionSpace(mesh, "CG", 1)

    up = Function(V)
    u, p = split(up)
    vq = Function(V)
    v, q = split(vq)
    c = Function(W)

    F = (
        inner(grad(u), grad(v)) * dx
        - p * div(v) * dx
        - q * div(u) * dx
        - dot(c, v) * dx
    )

    def pressure_point(x, on_boundary):
        return near(x[0], 0) and near(x[1], 0)

    bcs = cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0, 0)), geometry.boundaries, [1, 2, 3]
    )
    lid_velocity = Expression(("4*x[0]*(1-x[0])", "0.0"), degree=2)
    bcs += cashocs.create_dirichlet_bcs(V.sub(0), lid_velocity, geometry.boundaries, 4)
    bcs += [DirichletBC(V.sub(1), Constant(0), pressure_point, method="pointwise")]
    u_d = Expression(
        (
            "sqrt(pow(x[0], 2) + pow(x[1], 2))*cos(2*pi*x[1])",
            "-sqrt(pow(x[0], 2) + pow(x[1], 2))*sin(2*pi*x[0])",
        ),
        degree=2,
    )
    J = cashocs.IntegralFunctional(Constant(0.5) * dot(u - u_d, u - u_d) * dx)

    u_, p_ = TrialFunctions(V)
    v_, q_ = TestFunctions(V)
    pc_form = inner(grad(u_), grad(v_)) * dx + p_ * q_ * dx

    ksp_options = {
        "ksp_type": "minres",
        "ksp_max_it": 90,
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-30,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_0_pc_hypre_type": "boomeramg",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "jacobi",
    }

    ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J,
        up,
        c,
        vq,
        config=config_ocp,
        preconditioner_forms=pc_form,
        ksp_options=ksp_options,
    )
    ocp.solve(rtol=1e-2, max_iter=38)


def test_snes(F, bcs, J, y, u, p, config_ocp):
    config_ocp.set("StateSystem", "is_linear", "False")
    config_ocp.set("StateSystem", "backend", "petsc")

    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=17)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_pseudo_time_stepping(F, bcs, J, y, u, p, config_ocp):
    config_ocp.set("StateSystem", "is_linear", "False")
    config_ocp.set("StateSystem", "backend", "petsc")

    ksp_options = {
        "ts_type": "beuler",
        "ts_max_steps": 100,
        "ts_dt": 1e-1,
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config=config_ocp, ksp_options=ksp_options
    )
    ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=17)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_adjoint_linearizations(geometry, config_ocp):
    mesh = geometry.mesh
    dx = geometry.dx

    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, v_elem * p_elem)
    W = VectorFunctionSpace(mesh, "CG", 1)

    up = Function(V)
    u, p = split(up)
    vq = Function(V)
    v, q = split(vq)
    c = Function(W)

    F = (
        inner(grad(u), grad(v)) * dx
        + dot(grad(u) * u, v) * dx
        - p * div(v) * dx
        - q * div(u) * dx
        - dot(c, v) * dx
    )

    def pressure_point(x, on_boundary):
        return near(x[0], 0) and near(x[1], 0)

    bcs = cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0, 0)), geometry.boundaries, [1, 2, 3]
    )
    lid_velocity = Expression(("4*x[0]*(1-x[0])", "0.0"), degree=2)
    bcs += cashocs.create_dirichlet_bcs(V.sub(0), lid_velocity, geometry.boundaries, 4)
    bcs += [DirichletBC(V.sub(1), Constant(0), pressure_point, method="pointwise")]
    u_d = Expression(
        (
            "sqrt(pow(x[0], 2) + pow(x[1], 2))*cos(2*pi*x[1])",
            "-sqrt(pow(x[0], 2) + pow(x[1], 2))*sin(2*pi*x[0])",
        ),
        degree=2,
    )
    J = cashocs.IntegralFunctional(Constant(0.5) * dot(u - u_d, u - u_d) * dx)

    u_, p_ = TrialFunctions(V)
    v_, q_ = TestFunctions(V)
    dF = (
        inner(grad(u_), grad(v_)) * dx
        + dot(grad(u_) * u, v_) * dx
        - p_ * div(v_) * dx
        - q_ * div(u_) * dx
    )

    ksp_options = {
        "snes_type": "newtonls",
        "snes_max_it": 100,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_linesearch_type": "basic",
        "snes_rtol": 1e-6,
        "snes_atol": 1e-30,
        "ksp_type": "fgmres",
        "ksp_max_it": 2000,
        "ksp_rtol": 1e-3,
        "ksp_atol": 1e-30,
        "ksp_converged_reason": None,
        # "ksp_monitor_true_residual": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_0_pc_hypre_type": "boomeramg",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "jacobi",
    }

    config_ocp.set("StateSystem", "backend", "petsc")
    config_ocp.set("StateSystem", "is_linear", "False")
    config_ocp.set("StateSystem", "use_adjoint_linearizations", "True")

    ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J,
        up,
        c,
        vq,
        config=config_ocp,
        newton_linearizations=dF,
        ksp_options=ksp_options,
    )
    # ocp.compute_adjoint_variables()
    ocp.solve(rtol=1e-2, max_iter=47)
