# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
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

"""Basic line search algorithm with constant step size."""

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


class BasicLineSearch(line_search.LineSearch):
    """Implementation of the basic line search procedure with constant step size."""

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

        self.constant_stepsize = self.stepsize

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
        self.stepsize = self.constant_stepsize
        is_remeshed = False

        log.begin("Basic line search.", level=log.DEBUG)
        while True:
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

            objective_step = self._compute_objective_at_new_iterate()
            if objective_step is not None:
                log.debug(
                    f"Stepsize: {self.stepsize:.3e}  -"
                    f"  Tentative cost function value: {objective_step:.3e}"
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
            else:
                log.debug(
                    f"Stepsize: {self.stepsize:.3e} - Objective could not be evaluated."
                )
                self.stepsize /= self.beta_armijo
                self.optimization_variable_abstractions.revert_variable_update()

        log.end()
        solver.stepsize = self.stepsize

        if self.problem_type == "shape":
            return (self.optimization_variable_abstractions.deformation, is_remeshed)
        else:
            return (None, False)

    def _compute_objective_at_new_iterate(self) -> float | None:
        """Computes the objective value for the new (trial) iterate.

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
                objective_step = None
                self.state_problem.revert_to_checkpoint()

        return objective_step
