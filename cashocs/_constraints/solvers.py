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

"""Solvers for PDE constrained optimization problems with additional constraints."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List, Union, Optional

import fenics
import numpy as np
import ufl.core.expr

from cashocs import _loggers
from cashocs import utils
from cashocs._constraints import constraints

if TYPE_CHECKING:
    from cashocs._constraints import constrained_problems


class ConstrainedSolver(abc.ABC):
    """A solver for a constrained optimization problem."""

    solver_name: str

    def __init__(
        self,
        constrained_problem: constrained_problems.ConstrainedOptimizationProblem,
        mu_0: Optional[float] = None,
        lambda_0: Optional[List[float]] = None,
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
            if constraint.is_pointwise_constraint:
                mesh = constraint.measure.ufl_domain().ufl_cargo()
                self.CG = fenics.FunctionSpace(mesh, "CG", 1)
                break

        for i, constraint in enumerate(self.constraints):
            if constraint.is_pointwise_constraint:
                constraint.multiplier.vector().vec().set(self.lmbd[i])
                self.lmbd[i] = constraint.multiplier

        self.constraint_violation = 0.0
        self.constraint_violation_prev = 0.0
        self.beta = 10.0
        self.inner_cost_functional_shift = 0.0

        self.inner_scalar_tracking_forms = []
        self.inner_min_max_terms = []

    @abc.abstractmethod
    def _update_cost_functional(self) -> None:
        """Updates the cost functional with new weights."""
        pass

    def _post_process_cost_functional(self) -> None:
        """Ensures that scalar_tracking_forms and min_max_terms are correct."""
        if len(self.inner_scalar_tracking_forms) == 0:
            self.inner_scalar_tracking_forms = (
                self.constrained_problem.scalar_tracking_forms_initial
            )
        else:
            if self.constrained_problem.scalar_tracking_forms_initial is not None:
                self.inner_scalar_tracking_forms += (
                    self.constrained_problem.scalar_tracking_forms_initial
                )

        if len(self.inner_min_max_terms) == 0:
            self.inner_min_max_terms = None

    @abc.abstractmethod
    def solve(
        self,
        tol: float = 1e-2,
        max_iter: int = 25,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
        constraint_tol: Optional[float] = None,
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

    def print_results(self) -> None:
        """Prints the results of the current iteration to the console."""
        strs = [
            f"{self.solver_name} - Iteration {self.iterations:4d} -",
            f" Objective value: {self.constrained_problem.current_function_value:.3e}",
            f"    Constraint violation: {self.constraint_violation:.3e}",
            f"    Penalty parameter mu: {self.mu:.3e}",
        ]

        print("".join(strs))


class AugmentedLagrangianMethod(ConstrainedSolver):
    """An augmented Lagrangian method."""

    def __init__(
        self,
        constrained_problem: constrained_problems.ConstrainedOptimizationProblem,
        mu_0: Optional[float] = None,
        lambda_0: Optional[List[float]] = None,
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
        self.A_tensors = [fenics.PETScMatrix()] * self.constraint_dim
        self.b_tensors = [fenics.PETScVector()] * self.constraint_dim
        self.solver_name = "Augmented Lagrangian method"

    # noinspection PyPep8Naming
    def _project_pointwise_multiplier(
        self,
        project_terms: Union[ufl.core.expr.Expr, List[ufl.core.expr.Expr]],
        measure: fenics.Measure,
        multiplier: fenics.Function,
        A_tensor: fenics.PETScMatrix,
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
            project_term = utils.summation(project_terms)
        else:
            project_term = project_terms

        trial = fenics.TrialFunction(self.CG)
        test = fenics.TestFunction(self.CG)

        lhs = trial * test * measure
        rhs = project_term * test * measure

        utils._assemble_and_solve_linear(
            lhs, rhs, A=A_tensor, b=b_tensor, x=multiplier.vector().vec()
        )
        multiplier.vector().apply("")

    def _update_cost_functional(self) -> None:
        """Updates the cost functional with new weights."""
        self.inner_cost_functional_shifts = []

        self.inner_cost_functional_form = (
            self.constrained_problem.cost_functional_form_initial[:]
        )

        self.inner_scalar_tracking_forms = []
        self.inner_min_max_terms = []

        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint, constraints.EqualityConstraint):
                if constraint.is_integral_constraint:
                    self.inner_cost_functional_form += [
                        fenics.Constant(self.lmbd[i]) * constraint.linear_term
                    ]
                    self.inner_cost_functional_shifts.append(
                        -self.lmbd[i] * constraint.target
                    )

                    constraint.quadratic_term["weight"] = self.mu
                    self.inner_scalar_tracking_forms += [constraint.quadratic_term]

                elif constraint.is_pointwise_constraint:
                    self.inner_cost_functional_form += [
                        constraint.linear_term,
                        fenics.Constant(self.mu) * constraint.quadratic_term,
                    ]
                    self.inner_cost_functional_shifts.append(
                        -fenics.assemble(
                            self.lmbd[i] * constraint.target * constraint.measure
                        )
                    )

            elif isinstance(constraint, constraints.InequalityConstraint):
                if constraint.is_integral_constraint:
                    constraint.min_max_term["mu"] = self.mu
                    constraint.min_max_term["lambda"] = self.lmbd[i]
                    self.inner_min_max_terms += [constraint.min_max_term]

                elif constraint.is_pointwise_constraint:
                    constraint.weight.vector().vec().set(self.mu)
                    self.inner_cost_functional_form += constraint.cost_functional_terms

        self._post_process_cost_functional()
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
            # noinspection PyTypeChecker
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
                self.constraints[index].min_max_term["integrand"]
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
            self.constraints[index].min_max_term["lambda"] = self.lmbd[index]

        elif self.constraints[index].is_pointwise_constraint:
            project_terms = []
            if self.constraints[index].upper_bound is not None:
                project_terms.append(
                    utils._max(
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
                    utils._min(
                        self.lmbd[index]
                        + self.mu
                        * (
                            self.constraints[index].variable_function
                            - self.constraints[index].lower_bound
                        ),
                        fenics.Constant(0.0),
                    )
                )

            # noinspection PyTypeChecker
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
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
        constraint_tol: Optional[float] = None,
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

            _loggers.debug(f"mu = {self.mu}")
            _loggers.debug(f"lambda = {self.lmbd}")

            self._update_cost_functional()

            self.constrained_problem._solve_inner_problem(
                tol=tol, inner_rtol=inner_rtol, inner_atol=inner_atol
            )

            self._update_lagrange_multiplier_estimates()

            self.constraint_violation_prev = self.constraint_violation
            self.constraint_violation = (
                self.constrained_problem.total_constraint_violation()
            )

            self.print_results()

            if self.constraint_violation > self.gamma * self.constraint_violation_prev:
                self.mu *= self.beta

            if self.constraint_violation <= convergence_tol:
                print(f"{self.solver_name} converged successfully.\n")
                break

            if self.iterations >= max_iter:
                print(f"{self.solver_name} did not converge.\n")
                break


class QuadraticPenaltyMethod(ConstrainedSolver):
    """A quadratic penalty method."""

    def __init__(
        self,
        constrained_problem: constrained_problems.ConstrainedOptimizationProblem,
        mu_0: Optional[float] = None,
        lambda_0: Optional[list[float]] = None,
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
        self.solver_name = "Quadratic Penalty Method"

    def solve(
        self,
        tol: float = 1e-2,
        max_iter: int = 25,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
        constraint_tol: Optional[float] = None,
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

            _loggers.debug(f"mu = {self.mu}")

            self._update_cost_functional()

            self.constrained_problem._solve_inner_problem(tol=tol)

            self.constraint_violation = (
                self.constrained_problem.total_constraint_violation()
            )
            self.mu *= self.beta

            self.print_results()

            if self.constraint_violation <= convergence_tol:
                print(f"{self.solver_name} converged successfully.\n")
                break

            if self.iterations >= max_iter:
                print(f"{self.solver_name} did not converge.\n")
                break

    def _update_cost_functional(self) -> None:
        """Updates the cost functional with new weights."""
        self.inner_cost_functional_form = (
            self.constrained_problem.cost_functional_form_initial[:]
        )
        self.inner_scalar_tracking_forms = []
        self.inner_min_max_terms = []

        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint, constraints.EqualityConstraint):
                if constraint.is_integral_constraint:
                    constraint.quadratic_term["weight"] = self.mu
                    self.inner_scalar_tracking_forms += [constraint.quadratic_term]

                elif constraint.is_pointwise_constraint:
                    self.inner_cost_functional_form += [
                        fenics.Constant(self.mu) * constraint.quadratic_term,
                    ]

            elif isinstance(constraint, constraints.InequalityConstraint):
                if constraint.is_integral_constraint:
                    constraint.min_max_term["mu"] = self.mu
                    constraint.min_max_term["lambda"] = 0.0
                    self.inner_min_max_terms += [constraint.min_max_term]

                elif constraint.is_pointwise_constraint:
                    constraint.weight.vector().vec().set(self.mu)
                    constraint.multiplier.vector().vec().set(0.0)
                    self.inner_cost_functional_form += constraint.cost_functional_terms

        self._post_process_cost_functional()
