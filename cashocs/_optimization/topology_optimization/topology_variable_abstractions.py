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

"""Module for abstractions of optimization variables for topology optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import numpy as np

from cashocs._optimization import optimization_variable_abstractions

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import topology_optimization


class TopologyVariableAbstractions(
    optimization_variable_abstractions.OptimizationVariableAbstractions
):
    """Abstractions for optimization variables in the case of topology optimization."""

    def __init__(
        self,
        optimization_problem: topology_optimization.TopologyOptimizationProblem,
        db: database.Database,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            db: The database of the problem.

        """
        super().__init__(optimization_problem, db)

        self.levelset_function: fenics.Function = optimization_problem.levelset_function
        self.levelset_function_temp = fenics.Function(
            self.levelset_function.function_space()
        )
        self.optimization_problem = optimization_problem

        self.levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_temp.vector().apply("")

    def compute_decrease_measure(
        self, search_direction: list[fenics.Function] | None = None
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test.

        Args:
            search_direction: The search direction.

        Returns:
            The decrease measure for the Armijo test.

        """
        return 0.0

    def store_optimization_variables(self) -> None:
        """Saves a copy of the current iterate of the optimization variables."""
        self.levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_temp.vector().apply("")

    def revert_variable_update(self) -> None:
        """Reverts the optimization variables to the current iterate."""
        self.levelset_function.vector().vec().aypx(
            0.0, self.levelset_function_temp.vector().vec()
        )
        self.levelset_function.vector().apply("")

    def update_optimization_variables(
        self,
        search_direction: list[fenics.Function],
        stepsize: float,
        beta: float,
        active_idx: np.ndarray | None = None,
        constraint_gradient: np.ndarray | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> float:
        """Updates the optimization variables based on a line search.

        Args:
            search_direction: The current search direction.
            stepsize: The current (trial) stepsize.
            beta: The parameter for the line search, which "halves" the stepsize if the
                test was not successful.
            active_idx: The list of active indices of the working set. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.
            constraint_gradient: The gradient of the constraints for the mesh quality.
                Only needed for shape optimization with mesh quality constraints.
                Default is `None`.
            dropped_idx: The list of indicies for dropped constraints. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.

        Returns:
            The stepsize which was found to be acceptable.

        """
        self.store_optimization_variables()

        self.levelset_function.vector().vec().axpy(
            stepsize, search_direction[0].vector().vec()
        )
        self.levelset_function.vector().apply("")

        return stepsize

    def compute_gradient_norm(self) -> float:
        """Computes the norm of the gradient.

        Returns:
            The norm of the gradient.

        """
        return self.optimization_problem.solver.compute_angle()

    def compute_a_priori_decreases(
        self, search_direction: list[fenics.Function], stepsize: float
    ) -> int:
        """Computes the number of times the stepsize has to be "halved" a priori.

        Args:
            search_direction: The current search direction.
            stepsize: The current stepsize.

        Returns:
            The number of times the stepsize has to be "halved" before the actual trial.

        """
        return 0

    def requires_remeshing(self) -> bool:
        """Checks, if remeshing is needed.

        Returns:
            A boolean, which indicates whether remeshing is required.

        """
        return False

    def project_ncg_search_direction(
        self, search_direction: list[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """

    def compute_active_sets(self) -> None:
        """Computes the active sets of the problem."""
        pass
