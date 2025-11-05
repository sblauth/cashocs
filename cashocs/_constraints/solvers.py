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

"""Solvers for optimization problems with constraints."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _utils
from cashocs import log
from cashocs._constraints import constraints
from cashocs._optimization import cost_functional

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

if TYPE_CHECKING:
    try:
        from ufl_legacy.core import expr as ufl_expr
    except ImportError:
        from ufl.core import expr as ufl_expr

    from cashocs import _typing
    from cashocs._constraints import constrained_problems


class ConstrainedSolver(abc.ABC):
    """A solver for a constrained optimization problem."""

    solver_name: str

    def __init__(
        self,
        constrained_problem: constrained_problems.ConstrainedOptimizationProblem,
        mu_0: float | None = None,
        lambda_0: list[float] | float | None = None,
    ) -> None:
        """Initializes self.

        Args:
            constrained_problem: The constrained optimization problem which shall be
                solved.
            mu_0: Initial value for the penalty parameter, defaults to 1 (when ``None``
                is given).
            lambda_0: Initial guess for the Lagrange multipliers (in AugmentedLagrangian
                method) Defaults to zero initial guess, when ``None`` is given.

        """
        self.constrained_problem = constrained_problem

        self.constraints = self.constrained_problem.constraint_list
        self.constraint_dim = self.constrained_problem.constraint_dim
        self.output_manager = constrained_problem.output_manager
        self.iterations = 0

        if mu_0 is not None:
            self.mu = mu_0
        else:
            self.mu = 1.0

        if lambda_0 is not None:
            if not isinstance(lambda_0, list):
                self.lmbd = [lambda_0]
            else:
                self.lmbd = lambda_0
        else:
            self.lmbd = [0.0] * self.constraint_dim

        for constraint in self.constraints:
            if constraint.is_pointwise_constraint and constraint.measure is not None:
                mesh = constraint.measure.ufl_domain().ufl_cargo()
                self.cg_function_space = fenics.FunctionSpace(mesh, "CG", 1)
                break

        for i, constraint in enumerate(self.constraints):
            if constraint.is_pointwise_constraint:
                constraint.multiplier.vector().vec().set(self.lmbd[i])
                constraint.multiplier.vector().apply("")
                self.lmbd[i] = constraint.multiplier

        self.constraint_violation = 0.0
        self.constraint_violation_prev = 0.0
        self.beta = 10.0
        self.inner_cost_functional_shift = 0.0

    @abc.abstractmethod
    def _update_cost_functional(self) -> None:
        """Updates the cost functional with new weights."""
        pass

    @abc.abstractmethod
    def solve(
        self,
        tol: float = 1e-2,
        max_iter: int = 25,
        inner_rtol: float | None = None,
        inner_atol: float | None = None,
        constraint_tol: float | None = None,
    ) -> None:
        """Solves the constrained problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            max_iter: Maximum number of iterations for the outer solver. Default is 25.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            constraint_tol: The tolerance for the constraint violation, which is
                desired. If this is ``None`` (default), then this is specified as
                ``tol/10``.

        """
        pass

    def output(self) -> None:
        """Prints the results of the current iteration to the console."""
        db = self.constrained_problem.db
        optimization_state = db.parameter_db.optimization_state

        optimization_state["iteration"] = self.iterations - 1
        optimization_state["objective_value"] = (
            self.constrained_problem.current_function_value
        )
        optimization_state["constraint_violation"] = self.constraint_violation
        optimization_state["mu"] = self.mu

        self.output_manager.output()


class AugmentedLagrangianMethod(ConstrainedSolver):
    """An augmented Lagrangian method."""

    def __init__(
        self,
        constrained_problem: constrained_problems.ConstrainedOptimizationProblem,
        mu_0: float | None = None,
        lambda_0: list[float] | None = None,
    ) -> None:
        """Initializes self.

        Args:
            constrained_problem: The constrained optimization problem which shall be
                solved.
            mu_0: Initial value for the penalty parameter, defaults to 1 (when ``None``
                is given).
            lambda_0: Initial guess for the Lagrange multipliers (in AugmentedLagrangian
                method) Defaults to zero initial guess, when ``None`` is given.

        """
        super().__init__(constrained_problem, mu_0=mu_0, lambda_0=lambda_0)
        self.gamma = 0.25
        comm = constrained_problem.db.geometry_db.mpi_comm
        # pylint: disable=invalid-name
        self.A_tensors = [fenics.PETScMatrix(comm) for _ in range(self.constraint_dim)]
        self.b_tensors = [fenics.PETScVector(comm) for _ in range(self.constraint_dim)]
        self.solver_name = "Augmented Lagrangian"
        self.inner_cost_functional_form: list[_typing.CostFunctional] = []

    def _project_pointwise_multiplier(
        self,
        project_terms: ufl_expr.Expr | list[ufl_expr.Expr],
        measure: ufl.Measure,
        multiplier: fenics.Function,
        A_tensor: fenics.PETScMatrix,  # pylint: disable=invalid-name
        b_tensor: fenics.PETScVector,
    ) -> None:
        """Project the multiplier for a pointwise constraint to a FE function space.

        Args:
            project_terms: The ufl expression of the Lagrange multiplier (guess)
            measure: The measure, where the pointwise constraint is posed.
            multiplier: The function representing the Lagrange multiplier (guess)
            A_tensor: A matrix, into which the form is assembled for speed up
            b_tensor: A vector, into which the form is assembled for speed up

        """
        if isinstance(project_terms, list):
            project_term = _utils.summation(project_terms)
        else:
            project_term = project_terms

        trial = fenics.TrialFunction(self.cg_function_space)
        test = fenics.TestFunction(self.cg_function_space)

        lhs = trial * test * measure
        rhs = project_term * test * measure

        _utils.assemble_and_solve_linear(lhs, rhs, multiplier, A=A_tensor, b=b_tensor)

    def _update_cost_functional(self) -> None:
        """Updates the cost functional with new weights."""
        self.inner_cost_functional_shifts = []

        self.inner_cost_functional_form = (
            self.constrained_problem.cost_functional_form_initial[:]
        )

        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint, constraints.EqualityConstraint):
                if constraint.is_integral_constraint:
                    linear_functional = cost_functional.IntegralFunctional(
                        fenics.Constant(self.lmbd[i]) * constraint.linear_form
                    )
                    self.inner_cost_functional_form += [linear_functional]
                    self.inner_cost_functional_shifts.append(
                        -self.lmbd[i] * constraint.target
                    )

                    constraint.quadratic_functional.weight.assign(self.mu)
                    self.inner_cost_functional_form += [constraint.quadratic_functional]

                elif constraint.measure is not None:
                    quad_functional = cost_functional.IntegralFunctional(
                        fenics.Constant(self.mu) * constraint.quadratic_form
                    )
                    self.inner_cost_functional_form += [
                        constraint.linear_functional,
                        quad_functional,
                    ]
                    self.inner_cost_functional_shifts.append(
                        fenics.assemble(
                            -self.lmbd[i] * constraint.target * constraint.measure
                        )
                    )

            elif isinstance(constraint, constraints.InequalityConstraint):
                if constraint.is_integral_constraint:
                    constraint.min_max_term.mu.assign(self.mu)
                    constraint.min_max_term.lambd.assign(self.lmbd[i])
                    self.inner_cost_functional_form += [constraint.min_max_term]

                elif constraint.is_pointwise_constraint:
                    constraint.weight.assign(self.mu)
                    self.inner_cost_functional_form += constraint.cost_functional_terms

        self.inner_cost_functional_shift = np.sum(self.inner_cost_functional_shifts)

    def _update_equality_multipliers(self, index: int) -> None:
        """Performs an update of the Lagrange multipliers for equality constraints.

        Args:
            index: The index of the equality constraint.

        """
        if self.constraints[index].is_integral_constraint:
            self.lmbd[index] += self.mu * (
                fenics.assemble(self.constraints[index].variable_function)
                - self.constraints[index].target
            )
        elif self.constraints[index].is_pointwise_constraint:
            project_term = self.lmbd[index] + self.mu * (
                self.constraints[index].variable_function
                - self.constraints[index].target
            )
            self._project_pointwise_multiplier(
                project_term,
                self.constraints[index].measure,
                self.lmbd[index],
                self.A_tensors[index],
                self.b_tensors[index],
            )

    def _update_inequality_multipliers(self, index: int) -> None:
        """Performs an update of the Lagrange multipliers for equality constraints.

        Args:
            index: The index of the equality constraint.

        """
        if self.constraints[index].is_integral_constraint:
            lower_term = 0.0
            upper_term = 0.0

            min_max_integral = fenics.assemble(
                self.constraints[index].min_max_term.integrand
            )

            if self.constraints[index].lower_bound is not None:
                lower_term = np.minimum(
                    self.lmbd[index]
                    + self.mu
                    * (min_max_integral - self.constraints[index].lower_bound),
                    0.0,
                )

            if self.constraints[index].upper_bound is not None:
                upper_term = np.maximum(
                    self.lmbd[index]
                    + self.mu
                    * (min_max_integral - self.constraints[index].upper_bound),
                    0.0,
                )

            self.lmbd[index] = lower_term + upper_term
            self.constraints[index].min_max_term.lambd.assign(self.lmbd[index])

        elif self.constraints[index].is_pointwise_constraint:
            project_terms = []
            if self.constraints[index].upper_bound is not None:
                project_terms.append(
                    _utils.max_(
                        self.lmbd[index]
                        + self.mu
                        * (
                            self.constraints[index].variable_function
                            - self.constraints[index].upper_bound
                        ),
                        fenics.Constant(0.0),
                    )
                )

            if self.constraints[index].lower_bound is not None:
                project_terms.append(
                    _utils.min_(
                        self.lmbd[index]
                        + self.mu
                        * (
                            self.constraints[index].variable_function
                            - self.constraints[index].lower_bound
                        ),
                        fenics.Constant(0.0),
                    )
                )

            self._project_pointwise_multiplier(
                project_terms,
                self.constraints[index].measure,
                self.lmbd[index],
                self.A_tensors[index],
                self.b_tensors[index],
            )

    def _update_lagrange_multiplier_estimates(self) -> None:
        """Performs an update of the Lagrange multiplier estimates."""
        for i in range(self.constraint_dim):
            if isinstance(self.constraints[i], constraints.EqualityConstraint):
                self._update_equality_multipliers(i)

            elif isinstance(self.constraints[i], constraints.InequalityConstraint):
                self._update_inequality_multipliers(i)

    def solve(
        self,
        tol: float = 1e-2,
        max_iter: int = 10,
        inner_rtol: float | None = None,
        inner_atol: float | None = None,
        constraint_tol: float | None = None,
    ) -> None:
        """Solves the constrained problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            max_iter: Maximum number of iterations for the outer solver. Default is 25.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            constraint_tol: The tolerance for the constraint violation, which is
                desired. If this is ``None`` (default), then this is specified as
                ``tol/10``.

        """
        convergence_tol = constraint_tol or tol / 10.0

        self.iterations = 0
        while True:
            self.iterations += 1

            log.debug(f"mu = {self.mu:.3e}  lambda = {self.lmbd}")

            self._update_cost_functional()

            # pylint: disable=protected-access
            self.constrained_problem._solve_inner_problem(
                tol=tol,
                inner_rtol=inner_rtol,
                inner_atol=inner_atol,
                iteration=self.iterations,
            )

            self._update_lagrange_multiplier_estimates()

            self.constraint_violation_prev = self.constraint_violation
            self.constraint_violation = (
                self.constrained_problem.total_constraint_violation()
            )

            self.output()

            if self.constraint_violation > self.gamma * self.constraint_violation_prev:
                self.mu *= self.beta

            if self.constraint_violation <= convergence_tol:
                self.output_manager.output_summary()
                self.output_manager.post_process()
                break

            if self.iterations >= max_iter:
                self.output_manager.post_process()
                break


class QuadraticPenaltyMethod(ConstrainedSolver):
    """A quadratic penalty method."""

    def __init__(
        self,
        constrained_problem: constrained_problems.ConstrainedOptimizationProblem,
        mu_0: float | None = None,
        lambda_0: list[float] | None = None,
    ) -> None:
        """Initializes self.

        Args:
            constrained_problem: The constrained optimization problem which shall be
                solved.
            mu_0: Initial value for the penalty parameter, defaults to 1 (when ``None``
                is given).
            lambda_0: Initial guess for the Lagrange multipliers (in AugmentedLagrangian
                method) Defaults to zero initial guess, when ``None`` is given.

        """
        super().__init__(constrained_problem, mu_0=mu_0, lambda_0=lambda_0)
        self.solver_name = "Quadratic Penalty"

    def solve(
        self,
        tol: float = 1e-2,
        max_iter: int = 25,
        inner_rtol: float | None = None,
        inner_atol: float | None = None,
        constraint_tol: float | None = None,
    ) -> None:
        """Solves the constrained problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            max_iter: Maximum number of iterations for the outer solver. Default is 25.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            constraint_tol: The tolerance for the constraint violation, which is
                desired. If this is ``None`` (default), then this is specified as
                ``tol/10``.

        """
        convergence_tol = constraint_tol or tol / 10.0

        self.iterations = 0
        while True:
            self.iterations += 1

            log.debug(f"mu = {self.mu:.3e}")

            self._update_cost_functional()

            # pylint: disable=protected-access
            self.constrained_problem._solve_inner_problem(
                tol=tol,
                inner_rtol=inner_rtol,
                inner_atol=inner_atol,
                iteration=self.iterations,
            )

            self.constraint_violation = (
                self.constrained_problem.total_constraint_violation()
            )
            self.mu *= self.beta

            self.output()

            if self.constraint_violation <= convergence_tol:
                self.output_manager.output_summary()
                self.output_manager.post_process()
                break

            if self.iterations >= max_iter:
                self.output_manager.post_process()
                break

    def _update_cost_functional(self) -> None:
        """Updates the cost functional with new weights."""
        self.inner_cost_functional_form = (
            self.constrained_problem.cost_functional_form_initial[:]
        )

        for constraint in self.constraints:
            if isinstance(constraint, constraints.EqualityConstraint):
                if constraint.is_integral_constraint:
                    constraint.quadratic_functional.weight.assign(self.mu)
                    self.inner_cost_functional_form += [constraint.quadratic_functional]

                elif constraint.is_pointwise_constraint:
                    self.inner_cost_functional_form += [
                        cost_functional.IntegralFunctional(
                            fenics.Constant(self.mu) * constraint.quadratic_form
                        )
                    ]

            elif isinstance(constraint, constraints.InequalityConstraint):
                if constraint.is_integral_constraint:
                    constraint.min_max_term.mu.assign(self.mu)
                    constraint.min_max_term.lambd.assign(0.0)
                    self.inner_cost_functional_form += [constraint.min_max_term]

                elif constraint.is_pointwise_constraint:
                    constraint.weight.assign(self.mu)
                    constraint.multiplier.vector().vec().set(0.0)
                    constraint.multiplier.vector().apply("")
                    self.inner_cost_functional_form += constraint.cost_functional_terms
