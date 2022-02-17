# Copyright (C) 2020-2022 Sebastian Blauth
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

"""Tests for optimal control problems / algorithms.

"""

import os

from fenics import *
import numpy as np
import pytest

import cashocs
from cashocs._exceptions import InputError
from cashocs._exceptions import NotConvergedError

rng = np.random.RandomState(300696)
dir_path = os.path.dirname(os.path.realpath(__file__))
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx

ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)

cc = [0, 100]

ocp_cc = cashocs.OptimalControlProblem(
    F, bcs, J, y, u, p, config, control_constraints=cc
)


def test_control_constraints_handling():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    cg1_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    vcg1_elem = VectorElement("CG", mesh.ufl_cell(), 1)
    vcg2_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    real_elem = FiniteElement("R", mesh.ufl_cell(), 0)
    dg0_elem = FiniteElement("DG", mesh.ufl_cell(), 0)
    vdg0_elem = VectorElement("DG", mesh.ufl_cell(), 0)
    dg1_elem = FiniteElement("DG", mesh.ufl_cell(), 1)
    rt_elem = FiniteElement("RT", mesh.ufl_cell(), 1)

    mixed_elem = MixedElement([cg1_elem, dg0_elem, vdg0_elem])
    pass_elem = MixedElement([cg1_elem, real_elem, dg0_elem, vcg1_elem, mixed_elem])
    fail_elem1 = MixedElement([mixed_elem, cg1_elem, vdg0_elem, real_elem, rt_elem])
    fail_elem2 = MixedElement([dg1_elem, mixed_elem, cg1_elem, vdg0_elem, real_elem])
    fail_elem3 = MixedElement([mixed_elem, cg1_elem, vcg2_elem, vdg0_elem, real_elem])

    pass_space = FunctionSpace(mesh, pass_elem)
    pass_control = Function(pass_space)

    fail_space1 = FunctionSpace(mesh, fail_elem1)
    fail_space2 = FunctionSpace(mesh, fail_elem2)
    fail_space3 = FunctionSpace(mesh, fail_elem3)
    fail_control1 = Function(fail_space1)
    fail_control2 = Function(fail_space2)
    fail_control3 = Function(fail_space3)

    ocp_cc_pass = cashocs.OptimalControlProblem(
        F, bcs, J, y, pass_control, p, config, control_constraints=cc
    )

    with pytest.raises(InputError):
        ocp_cc_fail1 = cashocs.OptimalControlProblem(
            F, bcs, J, y, fail_control1, p, config, control_constraints=cc
        )
    with pytest.raises(InputError):
        ocp_cc_fail2 = cashocs.OptimalControlProblem(
            F, bcs, J, y, fail_control2, p, config, control_constraints=cc
        )
    with pytest.raises(InputError):
        ocp_cc_fail3 = cashocs.OptimalControlProblem(
            F, bcs, J, y, fail_control3, p, config, control_constraints=cc
        )


def test_control_gradient():
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, [u], rng=rng) > 1.9


def test_control_gd():
    u.vector()[:] = 0.0
    ocp._erase_pde_memory()
    ocp.solve("gd", rtol=1e-2, atol=0.0, max_iter=46)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_fr():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "FR")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=21)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_pr():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "PR")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=26)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_hs():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "HS")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=28)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_dy():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "DY")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=10)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_hz():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "HZ")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=28)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_restart_periodic():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "DY")
    config.set("AlgoCG", "cg_periodic_restart", "True")
    config.set("AlgoCG", "cg_periodic_its", "5")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=10)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_cg_restart_relative():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "DY")
    config.set("AlgoCG", "cg_relative_restart", "True")
    config.set("AlgoCG", "cg_restart_tol", "1.0")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("cg", rtol=1e-2, atol=0.0, max_iter=24)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_bfgs():
    u.vector()[:] = 0.0
    ocp._erase_pde_memory()
    ocp.solve("bfgs", rtol=1e-2, atol=0.0, max_iter=7)
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_control_newton_cg():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoTNM", "inner_newton", "cg")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("newton", rtol=1e-2, atol=0.0, max_iter=2)
    assert ocp.solver.relative_norm <= 1e-6


def test_control_newton_cr():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoTNM", "inner_newton", "cr")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve("newton", rtol=1e-2, atol=0.0, max_iter=2)
    assert ocp.solver.relative_norm <= 1e-6


def test_control_gd_cc():
    u.vector()[:] = 0.0
    ocp_cc._erase_pde_memory()
    ocp_cc.solve("gd", rtol=1e-2, atol=0.0, max_iter=22)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_cg_fr_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "FR")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("cg", rtol=1e-2, atol=0.0, max_iter=48)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_cg_pr_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "PR")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("cg", rtol=1e-2, atol=0.0, max_iter=25)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_cg_hs_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "HS")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("cg", rtol=1e-2, atol=0.0, max_iter=30)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_cg_dy_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "DY")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("cg", rtol=1e-2, atol=0.0, max_iter=9)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_cg_hz_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoCG", "cg_method", "HZ")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("cg", rtol=1e-2, atol=0.0, max_iter=37)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_lbfgs_cc():
    u.vector()[:] = 0.0
    ocp_cc._erase_pde_memory()
    ocp_cc.solve("lbfgs", rtol=1e-2, atol=0.0, max_iter=11)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_newton_cg_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoTNM", "inner_newton", "cg")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("newton", rtol=1e-2, atol=0.0, max_iter=8)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_control_newton_cr_cc():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("AlgoTNM", "inner_newton", "cr")
    u.vector()[:] = 0.0
    ocp_cc = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, control_constraints=cc
    )
    ocp_cc.solve("newton", rtol=1e-2, atol=0.0, max_iter=9)
    assert ocp_cc.solver.relative_norm <= ocp_cc.solver.rtol
    assert np.alltrue(ocp_cc.controls[0].vector()[:] >= cc[0])
    assert np.alltrue(ocp_cc.controls[0].vector()[:] <= cc[1])


def test_custom_supply_control():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 0.0
    adjoint_form = (
        inner(grad(p), grad(TestFunction(V))) * dx - (y - y_d) * TestFunction(V) * dx
    )
    dJ = Constant(alpha) * u * TestFunction(V) * dx + TestFunction(V) * p * dx

    user_ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    user_ocp.supply_custom_forms(dJ, adjoint_form, bcs)

    assert cashocs.verification.control_gradient_test(user_ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(user_ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(user_ocp, rng=rng) > 1.9

    user_ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    user_ocp.supply_custom_forms([dJ], [adjoint_form], [bcs])

    assert cashocs.verification.control_gradient_test(user_ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(user_ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(user_ocp, rng=rng) > 1.9


def test_scalar_norm_optimization():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 1e-3

    J = Constant(0) * dx
    norm_y = y * y * dx
    tracking_goal = rng.uniform(0.25, 0.75)
    J_norm = {"integrand": norm_y, "tracking_goal": tracking_goal}
    config.set("OptimizationRoutine", "initial_stepsize", "4e3")

    test_ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, scalar_tracking_forms=J_norm
    )
    test_ocp.solve(algorithm="bfgs", rtol=1e-3)

    assert 0.5 * pow(assemble(norm_y) - tracking_goal, 2) < 1e-15

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scalar_tracking_weight():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 1e-3

    J = Constant(0) * dx
    norm_y = y * y * dx
    tracking_goal = rng.uniform(0.25, 0.75)
    weight = rng.uniform(1.0, 1e3)
    J_norm = {"integrand": norm_y, "tracking_goal": tracking_goal, "weight": 1.0}
    config.set("OptimizationRoutine", "initial_stepsize", "4e3")

    test_ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, scalar_tracking_forms=J_norm
    )
    test_ocp.compute_state_variables()
    initial_function_value = 0.5 * pow(assemble(norm_y) - tracking_goal, 2)
    J_norm["weight"] = weight / initial_function_value

    test_ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, scalar_tracking_forms=J_norm
    )
    test_ocp.compute_state_variables()
    val = test_ocp.reduced_cost_functional.evaluate()
    assert np.abs(val - weight) < 1e-15

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scalar_multiple_norms():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 40

    J = Constant(0) * dx
    norm_y = y * y * dx
    norm_u = u * u * dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = {"integrand": norm_y, "tracking_goal": tracking_goals[0]}
    J_u = {"integrand": norm_u, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_y, J_u]
    config.set("OptimizationRoutine", "initial_stepsize", "1e-4")

    test_ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, scalar_tracking_forms=J_scalar
    )
    test_ocp.solve(algorithm="bfgs", rtol=1e-6, max_iter=500)

    assert 0.5 * pow(assemble(norm_y) - tracking_goals[0], 2) < 1e-2
    assert 0.5 * pow(assemble(norm_u) - tracking_goals[1], 2) < 1e-4

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_different_spaces():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

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

    J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * lambd) * u * u * dx

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    ocp.solve(algorithm="bfgs")
    assert ocp.solver.relative_norm <= ocp.solver.rtol


def test_nonlinear_state_eq():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    initial_guess = Function(V)
    F = inner(grad(y), grad(p)) * dx + pow(y, 3) * p * dx - u * p * dx
    config.set("StateSystem", "is_linear", "False")
    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, initial_guess=[initial_guess]
    )
    cashocs.verification.control_gradient_test(ocp, rng=rng)
    cashocs.verification.control_gradient_test(ocp, rng=rng)
    cashocs.verification.control_gradient_test(ocp, rng=rng)


def test_riesz_scalar_products():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    u.vector()[:] = 0.0
    riesz_scalar_product = TrialFunction(V) * TestFunction(V) * dx
    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, riesz_scalar_products=riesz_scalar_product
    )
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config, riesz_scalar_products=[riesz_scalar_product]
    )
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_hooks():
    def pre_function():
        u.vector()[:] = 1.0

    def post_function():
        u.vector()[:] = -1.0

    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)
    grad = Function(V)
    grad.vector()[:] = ocp.compute_gradient()[0].vector()[:]

    ocp.inject_pre_post_hook(pre_function, post_function)
    assert ocp.form_handler._pre_hook == pre_function
    assert ocp.form_handler._post_hook == post_function

    ocp.compute_state_variables()
    assert np.max(np.abs(u.vector()[:] - 1.0)) < 1e-15

    injected_grad = ocp.compute_gradient()
    assert np.max(np.abs(u.vector()[:] - (-1.0))) < 1e-15

    assert np.max(np.abs(grad.vector()[:] - injected_grad[0].vector()[:])) > 1e-3


def test_scaling_control():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 1e-2

    J1 = Constant(0.5) * (y - y_d) * (y - y_d) * dx
    J2 = Constant(0.5) * u * u * dx
    J_list = [J1, J2]

    desired_weights = rng.rand(2).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F, bcs, J_list, y, u, p, config, desired_weights=desired_weights
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14
    assert abs(assemble(J_list[0]) - desired_weights[0]) < 1e-14
    assert abs(assemble(J_list[1]) - desired_weights[1]) < 1e-14

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scaling_scalar_only():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 40

    J = Constant(0) * dx
    norm_y = y * y * dx
    norm_u = u * u * dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = {"integrand": norm_y, "tracking_goal": tracking_goals[0]}
    J_u = {"integrand": norm_u, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_y, J_u]

    desired_weights = rng.rand(3).tolist()
    summ = sum(desired_weights[1:])

    test_ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        [J],
        y,
        u,
        p,
        config,
        desired_weights=desired_weights,
        scalar_tracking_forms=J_scalar,
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scaling_scalar_and_single_cost():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 40

    norm_y = y * y * dx
    norm_u = u * u * dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = {"integrand": norm_y, "tracking_goal": tracking_goals[0]}
    J_u = {"integrand": norm_u, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_y, J_u]

    desired_weights = rng.rand(3).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        [J],
        y,
        u,
        p,
        config,
        desired_weights=desired_weights,
        scalar_tracking_forms=J_scalar,
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_scaling_all():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    u.vector()[:] = 40

    norm_y = y * y * dx
    norm_u = u * u * dx
    tracking_goals = [0.24154615814336944, 1554.0246268346273]
    J_y = {"integrand": norm_y, "tracking_goal": tracking_goals[0]}
    J_u = {"integrand": norm_u, "tracking_goal": tracking_goals[1]}
    J_scalar = [J_y, J_u]

    J1 = Constant(0.5) * (y - y_d) * (y - y_d) * dx
    J2 = Constant(0.5) * u * u * dx
    J_list = [J1, J2]

    desired_weights = rng.rand(4).tolist()
    summ = sum(desired_weights)

    test_ocp = cashocs.OptimalControlProblem(
        F,
        bcs,
        J_list,
        y,
        u,
        p,
        config,
        desired_weights=desired_weights,
        scalar_tracking_forms=J_scalar,
    )
    val = test_ocp.reduced_cost_functional.evaluate()

    assert abs(val - summ) < 1e-14

    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9
    assert cashocs.verification.control_gradient_test(ocp, rng=rng) > 1.9


def test_iterative_gradient():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")

    config.set("OptimizationRoutine", "gradient_method", "iterative")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9


def test_small_stepsize1():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    config.set("OptimizationRoutine", "initial_stepsize", "1e-8")
    u.vector()[:] = 0.0
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    with pytest.raises(NotConvergedError) as e_info:
        ocp.solve("gd", rtol=1e-2, atol=0.0, max_iter=2)
    assert "Armijo rule failed." in str(e_info.value)
