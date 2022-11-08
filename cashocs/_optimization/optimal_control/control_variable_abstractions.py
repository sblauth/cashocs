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

"""Management of control variables."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _utils
from cashocs._optimization import optimization_variable_abstractions

if TYPE_CHECKING:
    from cashocs._optimization import optimal_control


class ControlVariableAbstractions(
    optimization_variable_abstractions.OptimizationVariableAbstractions
):
    """Abstractions for optimization variables in the case of optimal control."""

    def __init__(
        self, optimization_problem: optimal_control.OptimalControlProblem
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.

        """
        super().__init__(optimization_problem)

        self.controls = optimization_problem.controls
        self.control_temp = _utils.create_function_list(
            optimization_problem.control_spaces
        )
        for i in range(len(self.controls)):
            self.control_temp[i].vector().vec().aypx(
                0.0, self.controls[i].vector().vec()
            )
            self.control_temp[i].vector().apply("")

        self.control_constraints = optimization_problem.control_constraints

        self.projected_difference = [
            fenics.Function(function_space)
            for function_space in self.form_handler.control_spaces
        ]

    def compute_decrease_measure(
        self, search_direction: Optional[List[fenics.Function]] = None
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test.

        Args:
            search_direction: The search direction.

        Returns:
            The decrease measure for the Armijo test.

        """
        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0,
                self.controls[j].vector().vec() - self.control_temp[j].vector().vec(),
            )
            self.projected_difference[j].vector().apply("")

        return self.form_handler.scalar_product(
            self.gradient, self.projected_difference
        )

    def store_optimization_variables(self) -> None:
        """Saves a copy of the current iterate of the optimization variables."""
        for i in range(len(self.controls)):
            self.control_temp[i].vector().vec().aypx(
                0.0, self.controls[i].vector().vec()
            )
            self.control_temp[i].vector().apply("")

    def revert_variable_update(self) -> None:
        """Reverts the optimization variables to the current iterate."""
        for i in range(len(self.controls)):
            self.controls[i].vector().vec().aypx(
                0.0, self.control_temp[i].vector().vec()
            )
            self.controls[i].vector().apply("")

    def update_optimization_variables(
        self, search_direction: List[fenics.Function], stepsize: float, beta: float
    ) -> float:
        """Updates the optimization variables based on a line search.

        Args:
            search_direction: The current search direction.
            stepsize: The current (trial) stepsize.
            beta: The parameter for the line search, which "halves" the stepsize if the
                test was not successful.

        Returns:
            The stepsize which was found to be acceptable.

        """
        self.store_optimization_variables()

        for j in range(len(self.controls)):
            self.controls[j].vector().vec().axpy(
                stepsize, search_direction[j].vector().vec()
            )
            self.controls[j].vector().apply("")

        self.form_handler.project_to_admissible_set(self.controls)

        return stepsize

    def compute_gradient_norm(self) -> float:
        """Computes the norm of the gradient.

        Returns:
            The norm of the gradient.

        """
        result: float = np.sqrt(self._stationary_measure_squared())
        return result

    def _stationary_measure_squared(self) -> float:
        """Computes the stationary measure (squared) corresponding to box-constraints.

        In case there are no box constraints this reduces to the classical gradient
        norm.

        Returns:
            The square of the stationary measure

        """
        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0, self.controls[j].vector().vec() - self.gradient[j].vector().vec()
            )
            self.projected_difference[j].vector().apply("")

        self.form_handler.project_to_admissible_set(self.projected_difference)

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0,
                self.controls[j].vector().vec()
                - self.projected_difference[j].vector().vec(),
            )
            self.projected_difference[j].vector().apply("")

        return self.form_handler.scalar_product(
            self.projected_difference, self.projected_difference
        )

    def compute_a_priori_decreases(
        self, search_direction: List[fenics.Function], stepsize: float
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
        self, search_direction: List[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """
        for j in range(self.form_handler.control_dim):
            idx = np.asarray(
                np.logical_or(
                    np.logical_and(
                        self.controls[j].vector()[:]
                        <= self.control_constraints[j][0].vector()[:],
                        search_direction[j].vector()[:] < 0.0,
                    ),
                    np.logical_and(
                        self.controls[j].vector()[:]
                        >= self.control_constraints[j][1].vector()[:],
                        search_direction[j].vector()[:] > 0.0,
                    ),
                )
            ).nonzero()[0]

            search_direction[j].vector()[idx] = 0.0
            search_direction[j].vector().apply("")
