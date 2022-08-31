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

import os

from fenics import *
import numpy as np

import cashocs

rng = np.random.RandomState(300696)
dir_path = os.path.dirname(os.path.realpath(__file__))

mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(8)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = dot(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])

J = pow(u - 0.5, 2) * dx


def test_int_eq_constraint():
    y.vector().vec().set(0.0)
    int_eq_target = rng.uniform(-1.0, 1.0)
    int_eq_constraint = cashocs.EqualityConstraint(y * y * dx, int_eq_target)
    assert int_eq_constraint.constraint_violation() == abs(int_eq_target)


def test_pw_eq_constraint():
    pw_eq_target = Function(V)
    pw_eq_target.vector().set_local(rng.random(pw_eq_target.vector().local_size()))
    pw_eq_target.vector().apply("")
    u.vector().set_local(rng.random(u.vector().local_size()))
    u.vector().apply("")
    pw_eq_constraint = cashocs.EqualityConstraint(u, pw_eq_target, dx)
    assert pw_eq_constraint.constraint_violation() == np.sqrt(
        assemble(pow(u - pw_eq_target, 2) * dx)
    )


def test_int_ineq_constraint():
    rng = np.random.RandomState(300696)

    bounds = rng.random(2)
    bounds.sort()
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    int_ineq_constraint_lower = cashocs.InequalityConstraint(
        u * dx, lower_bound=lower_bound
    )
    int_ineq_constraint_upper = cashocs.InequalityConstraint(
        u * dx, upper_bound=upper_bound
    )
    int_ineq_constraint_both = cashocs.InequalityConstraint(
        u * dx, lower_bound=lower_bound, upper_bound=upper_bound
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


def test_pw_ineq_constraint():
    lin_expr = Expression("2*(x[0] - 0.5)", degree=1)
    u.vector()[:] = interpolate(lin_expr, V).vector()[:]

    lower_bound = Function(V)
    lower_bound.vector().set_local(
        rng.uniform(0.0, 1.0, lower_bound.vector().local_size())
    )
    lower_bound.vector().apply("")
    upper_bound = Function(V)
    upper_bound.vector().set_local(
        rng.uniform(-1.0, 0.0, upper_bound.vector().local_size())
    )
    upper_bound.vector().apply("")

    ineq_constraint_lower = cashocs.InequalityConstraint(
        u, lower_bound=lower_bound, measure=dx
    )
    ineq_constraint_upper = cashocs.InequalityConstraint(
        u, upper_bound=upper_bound, measure=dx
    )
    ineq_constraint_both = cashocs.InequalityConstraint(
        u, lower_bound=lower_bound, upper_bound=upper_bound, measure=dx
    )

    assert ineq_constraint_lower.constraint_violation() == np.sqrt(
        assemble(pow(cashocs._utils.min_(0.0, u - lower_bound), 2) * dx)
    )

    assert ineq_constraint_upper.constraint_violation() == np.sqrt(
        assemble(pow(cashocs._utils.max_(0.0, u - upper_bound), 2) * dx)
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
                    * dx
                    + pow(cashocs._utils.max_(0.0, u - upper_bound), 2) * dx
                )
            )
        )
        < 1e-14
    )


def test_int_eq_constraints_only():
    u.vector()[:] = 1.0
    constraint = cashocs.EqualityConstraint(y * y * dx, 1.0)
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    cfg.set("Output", "verbose", "False")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2

    u.vector()[:] = 1.0
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2


def test_pw_eq_constraints_only():
    u.vector()[:] = 1.0
    constraint = cashocs.EqualityConstraint(y + u, 0.0, dx)
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    cfg.set("Output", "verbose", "False")
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2

    u.vector()[:] = 1.0
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-1)
    assert constraint.constraint_violation() < 1e-2


def test_int_ineq_constraints_only():
    u.vector()[:] = 0.0
    J = pow(y - Constant(1.0), 2) * dx
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    constraint = cashocs.InequalityConstraint(y * y * dx, upper_bound=0.5)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    J = pow(y - Constant(0.1), 2) * dx
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    constraint = cashocs.InequalityConstraint(y * y * dx, lower_bound=0.5)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3


def test_pw_ineq_constraints_only():
    u.vector()[:] = 0.0
    J = pow(y - Constant(1.0), 2) * dx
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    cfg.set("OptimizationRoutine", "maximum_iterations", "250")
    constraint = cashocs.InequalityConstraint(y, upper_bound=0.5, measure=dx)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    J = pow(y - Constant(-1), 2) * dx
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    cfg.set("OptimizationRoutine", "maximum_iterations", "250")
    constraint = cashocs.InequalityConstraint(y, lower_bound=-0.5, measure=dx)
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    lin_expr = Expression("2*(x[0] - 0.5)", degree=1)
    u.vector()[:] = 0.0
    J = pow(y - lin_expr, 2) * dx
    cfg = cashocs.load_config(dir_path + "/config_ocp.ini")
    cfg.set("OptimizationRoutine", "maximum_iterations", "250")
    constraint = cashocs.InequalityConstraint(
        y, lower_bound=-0.5, upper_bound=0.5, measure=dx
    )
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="AL", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3

    u.vector()[:] = 0.0
    problem = cashocs.ConstrainedOptimalControlProblem(
        F, bcs, J, y, u, p, constraint, config=cfg
    )
    problem.solve(method="QP", tol=1e-2)
    assert constraint.constraint_violation() <= 1e-3
