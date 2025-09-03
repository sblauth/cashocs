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

"""Polynomial line search algorithm."""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _exceptions
from cashocs import log
from cashocs._optimization.line_search import line_search

if TYPE_CHECKING:
    from scipy import sparse

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

        self.armijo_stepsize_initial = self.stepsize
        self.decrease_measure_w_o_step = 1.0

        self.factor_low = self.config.getfloat("LineSearch", "factor_low")
        self.factor_high = self.config.getfloat("LineSearch", "factor_high")
        self.polynomial_model = self.config.get(
            "LineSearch", "polynomial_model"
        ).casefold()
        self.f_vals: collections.deque[float] = collections.deque()
        self.alpha_vals: collections.deque[float] = collections.deque()

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
        if solver.iteration >= solver.max_iter:
            return True

        if self.stepsize * self.search_direction_inf <= 1e-8:
            log.error("Stepsize too small.")
            solver.line_search_broken = True
            return True
        elif (
            not self.is_newton_like
            and not self.is_newton
            and self.stepsize / self.armijo_stepsize_initial <= 1e-8
        ):
            log.error("Stepsize too small.")
            solver.line_search_broken = True
            return True

        return False

    def search(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: list[fenics.Function],
        has_curvature_info: bool,
        active_idx: np.ndarray | None = None,
        constraint_gradient: sparse.csr_matrix | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> tuple[fenics.Function | None, bool]:
        """Performs the line search.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates whether the direction is
                (presumably) scaled.
            active_idx: The list of active indices of the working set. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.
            constraint_gradient: The gradient of the constraints for the mesh quality.
                Only needed for shape optimization with mesh quality constraints.
                Default is `None`.
            dropped_idx: The list of indicies for dropped constraints. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.

        Returns:
            The accepted deformation / update or None, in case the update was not
            successful.

        """
        self.initialize_stepsize(solver, search_direction, has_curvature_info)

        self.f_vals.clear()
        self.alpha_vals.clear()

        is_remeshed = False
        log.begin("Polynomial line search.", level=log.DEBUG)
        while True:
            if self._check_for_nonconvergence(solver):
                return (None, False)

            if self.problem_type == "shape":
                self.decrease_measure_w_o_step = (
                    self.optimization_variable_abstractions.compute_decrease_measure(
                        search_direction
                    )
                )
            self.stepsize = (
                self.optimization_variable_abstractions.update_optimization_variables(
                    search_direction,
                    self.stepsize,
                    self.beta_armijo,
                    active_idx,
                    constraint_gradient,
                    dropped_idx,
                )
            )
            self.alpha_vals.append(self.stepsize)

            current_function_value = solver.objective_value

            self.state_problem.has_solution = False
            objective_step = self.cost_functional.evaluate()
            self.f_vals.append(objective_step)
            log.debug(
                f"Stepsize: {self.stepsize:.3e}  -"
                f"  Tentative cost function value: {objective_step:.3e}"
            )

            decrease_measure = self._compute_decrease_measure(search_direction)

            if self._satisfies_armijo_condition(
                objective_step, current_function_value, decrease_measure
            ):
                log.debug(
                    "Accepting tentative step based on Armijo decrease condition."
                )
                if self.optimization_variable_abstractions.requires_remeshing():
                    log.debug(
                        "The mesh quality was sufficient for accepting the step, "
                        "but the mesh cannot be used anymore for computing a gradient."
                        "Performing a remeshing operation."
                    )
                    is_remeshed = (
                        self.optimization_variable_abstractions.mesh_handler.remesh(
                            solver
                        )
                    )
                    break

                if solver.iteration == 0:
                    self.armijo_stepsize_initial = self.stepsize
                break

            else:
                log.debug(
                    "Rejecting tentative step based on Armijo decrease condition."
                )
                self.stepsize = self._compute_polynomial_stepsize(
                    current_function_value,
                    decrease_measure,
                    self.f_vals,
                    self.alpha_vals,
                )
                self.optimization_variable_abstractions.revert_variable_update()

        solver.stepsize = self.stepsize
        log.end()

        if not has_curvature_info:
            self.stepsize /= self.factor_high

        if self.problem_type == "shape":
            return (self.optimization_variable_abstractions.deformation, is_remeshed)
        else:
            return (None, False)

    def _compute_polynomial_stepsize(
        self,
        f_current: float,
        decrease_measure: float,
        f_vals: collections.deque[float],
        alpha_vals: collections.deque[float],
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
        f_vals: collections.deque[float],
        alpha_vals: collections.deque[float],
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
        f_vals: collections.deque[float],
        alpha_vals: collections.deque[float],
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
                    [alpha_vals[0] ** 2, -(alpha_vals[1] ** 2)],
                    [-(alpha_vals[0] ** 3), alpha_vals[1] ** 3],
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
        self, search_direction: list[fenics.Function]
    ) -> float:
        """Computes the decrease measure for use in the Armijo line search.

        Args:
            search_direction: The current search direction.

        Returns:
            The computed decrease measure.

        """
        if self.problem_type in ["control", "topology"]:
            return self.optimization_variable_abstractions.compute_decrease_measure(
                search_direction
            )
        elif self.problem_type == "shape":
            return self.decrease_measure_w_o_step * self.stepsize
        else:
            return float("inf")
