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

"""Armijo line search algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics

from cashocs import _exceptions
from cashocs import log
from cashocs._optimization.line_search import line_search

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse

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
            A tuple (defo, is_remeshed), where defo is accepted deformation / update
            or None, in case the update was not successful and is_remeshed is a boolean
            indicating whether a remeshing has been performed in the line search.

        """
        self.initialize_stepsize(solver, search_direction, has_curvature_info)
        is_remeshed = False

        log.begin("Armijo line search.", level=log.DEBUG)
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

            current_function_value = solver.objective_value
            objective_step = self._compute_objective_at_new_iterate(
                current_function_value
            )
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
                self.stepsize /= self.beta_armijo
                self.optimization_variable_abstractions.revert_variable_update()

        log.end()
        solver.stepsize = self.stepsize

        if not has_curvature_info:
            self.stepsize *= self.beta_armijo

        if self.problem_type == "shape":
            return (self.optimization_variable_abstractions.deformation, is_remeshed)
        else:
            return (None, False)

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

    def _compute_objective_at_new_iterate(self, current_function_value: float) -> float:
        """Computes the objective value for the new (trial) iterate.

        Args:
            current_function_value: The current function value.

        Returns:
            The value of the cost functional at the new iterate.

        """
        self.state_problem.has_solution = False
        try:
            objective_step = self.cost_functional.evaluate()
        except (_exceptions.PETScError, _exceptions.NotConvergedError) as error:
            if self.config.getboolean("LineSearch", "fail_if_not_converged"):
                raise error
            else:
                objective_step = 2.0 * abs(current_function_value)
                self.state_problem.revert_to_checkpoint()

        return objective_step
