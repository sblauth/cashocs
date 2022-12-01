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

from collections import namedtuple
import pathlib

from fenics import *
import numpy as np
import pytest

import cashocs


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
        CG1, Constant(0.0), geometry.boundaries, [1, 2, 3, 4]
    )


@pytest.fixture
def J(u, geometry):
    return cashocs.IntegralFunctional(pow(u - 0.5, 2) * geometry.dx)


def test_int_eq_constraint(rng, geometry, y):
    int_eq_target = rng.uniform(-1.0, 1.0)
    int_eq_constraint = cashocs.EqualityConstraint(y * y * geometry.dx, int_eq_target)
    assert int_eq_constraint.constraint_violation() == abs(int_eq_target)


def test_pw_eq_constraint(rng, CG1, u, geometry):
    pw_eq_target = Function(CG1)
    pw_eq_target.vector().set_local(rng.random(pw_eq_target.vector().local_size()))
    pw_eq_target.vector().apply("")
    u.vector().set_local(rng.random(u.vector().local_size()))
    u.vector().apply("")
    pw_eq_constraint = cashocs.EqualityConstraint(u, pw_eq_target, geometry.dx)
    assert pw_eq_constraint.constraint_violation() == np.sqrt(
        assemble(pow(u - pw_eq_target, 2) * geometry.dx)
    )


def test_int_ineq_constraint(rng, u, geometry):
    bounds = rng.random(2)
    bounds.sort()
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    int_ineq_constraint_lower = cashocs.InequalityConstraint(
        u * geometry.dx, lower_bound=lower_bound
    )
    int_ineq_constraint_upper = cashocs.InequalityConstraint(
        u * geometry.dx, upper_bound=upper_bound
    )
    int_ineq_constraint_both = cashocs.InequalityConstraint(
        u * geometry.dx, lower_bound=lower_bound, upper_bound=upper_bound
    )

    shift_tol = rng.random()
    u.vector().vec().set(lower_bound - shift_tol)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_lower.constraint_violation() - shift_tol) < 1e-14

    shift_tol = rng.random()
    u.vector().vec().set(lower_bound + shift_tol)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_lower.constraint_violation() - 0.0) < 1e-14

    shift_tol = rng.random()
    u.vector().vec().set(upper_bound + shift_tol)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_upper.constraint_violation() - shift_tol) < 1e-14

    shift_tol = rng.random()
    u.vector().vec().set(upper_bound - shift_tol)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_upper.constraint_violation() - 0.0) < 1e-14

    u.vector().vec().set((upper_bound + lower_bound) / 2.0)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_both.constraint_violation() - 0.0) < 1e-14

    shift_tol = rng.random()
    u.vector().vec().set(upper_bound + shift_tol)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_both.constraint_violation() - shift_tol) < 1e-14

    shift_tol = rng.random()
    u.vector().vec().set(lower_bound - shift_tol)
    u.vector().apply("")
    assert np.abs(int_ineq_constraint_both.constraint_violation() - shift_tol) < 1e-14


def test_pw_ineq_constraint(rng, u, CG1, geometry):
    lin_expr = Expression("2*(x[0] - 0.5)", degree=1)
    u.vector().vec().aypx(0.0, interpolate(lin_expr, CG1).vector().vec())
    u.vector().apply("")

    lower_bound = Function(CG1)
    lower_bound.vector().set_local(
        rng.uniform(0.0, 1.0, lower_bound.vector().local_size())
    )
    lower_bound.vector().apply("")
    upper_bound = Function(CG1)
    upper_bound.vector().set_local(
        rng.uniform(-1.0, 0.0, upper_bound.vector().local_size())
    )
    upper_bound.vector().apply("")

    ineq_constraint_lower = cashocs.InequalityConstraint(
        u, lower_bound=lower_bound, measure=geometry.dx
    )
    ineq_constraint_upper = cashocs.InequalityConstraint(
        u, upper_bound=upper_bound, measure=geometry.dx
    )
    ineq_constraint_both = cashocs.InequalityConstraint(
        u, lower_bound=lower_bound, upper_bound=upper_bound, measure=geometry.dx
    )

    assert ineq_constraint_lower.constraint_violation() == np.sqrt(
        assemble(pow(cashocs._utils.min_(0.0, u - lower_bound), 2) * geometry.dx)
    )

    assert ineq_constraint_upper.constraint_violation() == np.sqrt(
        assemble(pow(cashocs._utils.max_(0.0, u - upper_bound), 2) * geometry.dx)
    )

    assert (
        np.abs(
            ineq_constraint_both.constraint_violation()
            - np.sqrt(
                assemble(
                    pow(
                        cashocs._utils.min_(0.0, u - lower_bound),
                        2,
                    )
                    * geometry.dx
                    + pow(cashocs._utils.max_(0.0, u - upper_bound), 2) * geometry.dx
                )
            )
        )
        < 1e-14
    )


def test_int_eq_constraints_only(dir_path, u, y, geometry, F, bcs, J, p):
    u.vector().vec().set(1.0)
    u.vector().apply("")
    constraint = cashocs.EqualityConstraint(y * y * geometry.dx, 1.0)
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    cfg.set("Output", "verbose", "False")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2

    u.vector().vec().set(1.0)
    u.vector().apply("")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2


def test_pw_eq_constraints_only(geometry, F, bcs, J, y, u, p, config_ocp):
    u.vector().vec().set(1.0)
    u.vector().apply("")
    constraint = cashocs.EqualityConstraint(y + u, 0.0, geometry.dx)
    config_ocp.set("Output", "verbose", "False")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="AL", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2

    u.vector().vec().set(1.0)
    u.vector().apply("")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="QP", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2


def test_int_ineq_constraints_only(dir_path, config_ocp, geometry, F, bcs, y, u, p):
    u.vector().vec().set(0.0)
    u.vector().apply("")
    J = cashocs.IntegralFunctional(pow(y - Constant(1.0), 2) * geometry.dx)
    constraint = cashocs.InequalityConstraint(y * y * geometry.dx, upper_bound=0.5)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    J = cashocs.IntegralFunctional(pow(y - Constant(0.1), 2) * geometry.dx)
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    constraint = cashocs.InequalityConstraint(y * y * geometry.dx, lower_bound=0.5)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3


def test_pw_ineq_constraints_only(config_ocp, geometry, F, bcs, y, u, p):
    u.vector().vec().set(0.0)
    u.vector().apply("")
    J = cashocs.IntegralFunctional(pow(y - Constant(1.0), 2) * geometry.dx)
    config_ocp.set("OptimizationRoutine", "maximum_iterations", "500")
    constraint = cashocs.InequalityConstraint(y, upper_bound=0.5, measure=geometry.dx)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    J = cashocs.IntegralFunctional(pow(y - Constant(-1), 2) * geometry.dx)
    config_ocp.set("OptimizationRoutine", "maximum_iterations", "500")
    constraint = cashocs.InequalityConstraint(y, lower_bound=-0.5, measure=geometry.dx)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    lin_expr = Expression("2*(x[0] - 0.5)", degree=1)
    u.vector().vec().set(0.0)
    u.vector().apply("")
    J = cashocs.IntegralFunctional(pow(y - lin_expr, 2) * geometry.dx)
    config_ocp.set("OptimizationRoutine", "maximum_iterations", "500")
    constraint = cashocs.InequalityConstraint(
        y, lower_bound=-0.5, upper_bound=0.5, measure=geometry.dx
    )
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector().vec().set(0.0)
    u.vector().apply("")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=config_ocp
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3
