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

"""Armijo line search algorithm."""

from __future__ import annotations

from typing import List

import fenics
from typing_extensions import TYPE_CHECKING

from cashocs import _loggers
from cashocs._optimization.line_search import line_search

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import optimization_algorithms


class ArmijoLineSearch(line_search.LineSearch):
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

            current_function_value = solver.objective_value

            self.state_problem.has_solution = False
            objective_step = self.cost_functional.evaluate()

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
                self.stepsize /= self.beta_armijo
                self.optimization_variable_abstractions.revert_variable_update()

        solver.stepsize = self.stepsize

        if not has_curvature_info:
            self.stepsize *= self.beta_armijo

        return None

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
