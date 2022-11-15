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

"""Polynomial line search algorithm."""


from __future__ import annotations

import collections
from typing import Deque, List

import fenics
import numpy as np
from typing_extensions import TYPE_CHECKING

from cashocs import _exceptions
from cashocs import _loggers
from cashocs._optimization.line_search import line_search

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import optimization_algorithms


class PolynomialLineSearch(line_search.LineSearch):
    """Implementation of the Armijo line search procedure."""

    def __init__(
        self,
        db: database.Database,
        optimization_problem: _typing.OptimizationProblem,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            optimization_problem: The corresponding optimization problem.

        """
        super().__init__(db, optimization_problem)

        self.epsilon_armijo: float = self.config.getfloat(
            "LineSearch", "epsilon_armijo"
        )
        self.armijo_stepsize_initial = self.stepsize
        self.decrease_measure_w_o_step = 1.0

        self.factor_low = self.config.getfloat("LineSearch", "factor_low")
        self.factor_high = self.config.getfloat("LineSearch", "factor_high")
        self.polynomial_model = self.config.get(
            "LineSearch", "polynomial_model"
        ).casefold()
        self.f_vals: Deque[float] = collections.deque()
        self.alpha_vals: Deque[float] = collections.deque()

    def _check_for_nonconvergence(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> bool:
        """Checks, whether the line search failed to converge.

        Args:
            solver: The optimization algorithm, which uses the line search.

        Returns:
            A boolean, which is True if a termination / cancellation criterion is
            satisfied.

        """
        if solver.iteration >= solver.maximum_iterations:
            return True

        if self.stepsize * self.search_direction_inf <= 1e-8:
            _loggers.error("Stepsize too small.")
            solver.line_search_broken = True
            return True
        elif (
            not self.is_newton_like
            and not self.is_newton
            and self.stepsize / self.armijo_stepsize_initial <= 1e-8
        ):
            _loggers.error("Stepsize too small.")
            solver.line_search_broken = True
            return True

        return False

    def search(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
    ) -> None:
        """Performs the line search.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates whether the direction is
                (presumably) scaled.

        """
        self.initialize_stepsize(solver, search_direction, has_curvature_info)

        self.f_vals.clear()
        self.alpha_vals.clear()
        while True:
            if self._check_for_nonconvergence(solver):
                return None

            if self.db.parameter_db.problem_type == "shape":
                self.decrease_measure_w_o_step = (
                    self.optimization_variable_abstractions.compute_decrease_measure(
                        search_direction
                    )
                )
            self.stepsize = (
                self.optimization_variable_abstractions.update_optimization_variables(
                    search_direction, self.stepsize, self.beta_armijo
                )
            )
            self.alpha_vals.append(self.stepsize)

            current_function_value = solver.objective_value

            self.state_problem.has_solution = False
            objective_step = self.cost_functional.evaluate()
            self.f_vals.append(objective_step)

            decrease_measure = self._compute_decrease_measure(search_direction)

            if (
                objective_step
                < current_function_value + self.epsilon_armijo * decrease_measure
            ):
                if self.optimization_variable_abstractions.requires_remeshing():
                    self.optimization_variable_abstractions.mesh_handler.remesh(solver)
                    break

                if solver.iteration == 0:
                    self.armijo_stepsize_initial = self.stepsize
                break

            else:
                self.stepsize = self._compute_polynomial_stepsize(
                    current_function_value,
                    decrease_measure,
                    self.f_vals,
                    self.alpha_vals,
                )
                self.optimization_variable_abstractions.revert_variable_update()

        solver.stepsize = self.stepsize

        if not has_curvature_info:
            self.stepsize /= self.factor_high

        return None

    def _compute_polynomial_stepsize(
        self,
        f_current: float,
        decrease_measure: float,
        f_vals: Deque[float],
        alpha_vals: Deque[float],
    ) -> float:
        """Computes a stepsize based on polynomial models.

        Args:
            f_current: Current function value
            decrease_measure: Current directional derivative in descent direction
            f_vals: History of trial function values
            alpha_vals: History of trial stepsizes

        Returns:
            A new stepsize based on polynomial interpolation.

        """
        if len(f_vals) == 1:
            alpha = self._quadratic_stepsize_model(
                f_current, decrease_measure, f_vals, alpha_vals
            )
        elif len(f_vals) == 2:
            alpha = self._cubic_stepsize_model(
                f_current, decrease_measure, f_vals, alpha_vals
            )
        else:
            raise _exceptions.CashocsException("This code should not be reached.")

        if alpha < self.factor_low * alpha_vals[-1]:
            stepsize = self.factor_low * alpha_vals[-1]
        elif alpha > self.factor_high * alpha_vals[-1]:
            stepsize = self.factor_high * alpha_vals[-1]
        else:
            stepsize = alpha

        if (self.polynomial_model == "quadratic" and len(f_vals) == 1) or (
            self.polynomial_model == "cubic" and len(f_vals) == 2
        ):
            f_vals.popleft()
            alpha_vals.popleft()

        return float(stepsize)

    def _quadratic_stepsize_model(
        self,
        f_current: float,
        decrease_measure: float,
        f_vals: Deque[float],
        alpha_vals: Deque[float],
    ) -> float:
        """Computes a trial stepsize based on a quadratic model.

        Args:
            f_current: Current function value
            decrease_measure: Current directional derivative in descent direction
            f_vals: History of trial function values
            alpha_vals: History of trial stepsizes

        Returns:
            A new trial stepsize based on quadratic interpolation

        """
        stepsize = -(decrease_measure * self.stepsize**2) / (
            2.0 * (f_vals[0] - f_current - decrease_measure * alpha_vals[0])
        )
        return stepsize

    def _cubic_stepsize_model(
        self,
        f_current: float,
        decrease_measure: float,
        f_vals: Deque[float],
        alpha_vals: Deque[float],
    ) -> float:
        """Computes a trial stepsize based on a cubic model.

        Args:
            f_current: Current function value
            decrease_measure: Current directional derivative in descent direction
            f_vals: History of trial function values
            alpha_vals: History of trial stepsizes

        Returns:
            A new trial stepsize based on cubic interpolation

        """
        coeffs = (
            1
            / (
                alpha_vals[0] ** 2
                * alpha_vals[1] ** 2
                * (alpha_vals[1] - alpha_vals[0])
            )
            * np.array(
                [
                    [alpha_vals[0] ** 2, -alpha_vals[1] ** 2],
                    [-alpha_vals[0] ** 3, alpha_vals[1] ** 3],
                ]
            )
            @ np.array(
                [
                    f_vals[1] - f_current - decrease_measure * alpha_vals[1],
                    f_vals[0] - f_current - decrease_measure * alpha_vals[0],
                ]
            )
        )
        a, b = coeffs
        stepsize = (-b + np.sqrt(b**2 - 3 * a * decrease_measure)) / (3 * a)
        return float(stepsize)

    def _compute_decrease_measure(
        self, search_direction: List[fenics.Function]
    ) -> float:
        """Computes the decrease measure for use in the Armijo line search.

        Args:
            search_direction: The current search direction.

        Returns:
            The computed decrease measure.

        """
        if self.db.parameter_db.problem_type == "control":
            return self.optimization_variable_abstractions.compute_decrease_measure(
                search_direction
            )
        elif self.db.parameter_db.problem_type == "shape":
            return self.decrease_measure_w_o_step * self.stepsize
        else:
            return float("inf")
