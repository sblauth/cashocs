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

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import fenics
import numpy as np

from ..optimization_variable_handler import OptimizationVariableHandler


if TYPE_CHECKING:
    from .optimal_control_problem import OptimalControlProblem


class ControlVariableHandler(OptimizationVariableHandler):
    def __init__(self, optimization_problem: OptimalControlProblem) -> None:

        super().__init__(optimization_problem)

        self.controls = optimization_problem.controls
        self.control_temp = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]
        for i in range(len(self.controls)):
            self.control_temp[i].vector().vec().aypx(
                0.0, self.controls[i].vector().vec()
            )

        self.control_constraints = optimization_problem.control_constraints

        self.projected_difference = [
            fenics.Function(V) for V in self.form_handler.control_spaces
        ]

    def compute_decrease_measure(
        self, search_direction: Optional[List[fenics.Function]] = None
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test

        Parameters
        ----------
        search_direction : list[fenics.Function] or None, optional
            The search direction (not required)

        Returns
        -------
        float
            The decrease measure for the Armijo test
        """

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0,
                self.controls[j].vector().vec() - self.control_temp[j].vector().vec(),
            )

        return self.form_handler.scalar_product(
            self.gradient, self.projected_difference
        )

    def store_optimization_variables(self) -> None:

        for i in range(len(self.controls)):
            self.control_temp[i].vector().vec().aypx(
                0.0, self.controls[i].vector().vec()
            )

    def revert_variable_update(self) -> None:

        for i in range(len(self.controls)):
            self.controls[i].vector().vec().aypx(
                0.0, self.control_temp[i].vector().vec()
            )

    def update_optimization_variables(
        self, search_direction, stepsize: float, beta: float
    ) -> float:

        self.store_optimization_variables()

        for j in range(len(self.controls)):
            self.controls[j].vector().vec().axpy(
                stepsize, search_direction[j].vector().vec()
            )

        self.form_handler.project_to_admissible_set(self.controls)

        return stepsize

    def compute_gradient_norm(self) -> float:

        return np.sqrt(self._stationary_measure_squared())

    def _stationary_measure_squared(self) -> float:
        """Computes the stationary measure (squared) corresponding to box-constraints

        In case there are no box constraints this reduces to the classical gradient
        norm.

        Returns
        -------
         float
            The square of the stationary measure

        """

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0, self.controls[j].vector().vec() - self.gradient[j].vector().vec()
            )

        self.form_handler.project_to_admissible_set(self.projected_difference)

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0,
                self.controls[j].vector().vec()
                - self.projected_difference[j].vector().vec(),
            )

        return self.form_handler.scalar_product(
            self.projected_difference, self.projected_difference
        )

    def compute_a_priori_decreases(
        self, search_direction: List[fenics.Function], stepsize: float
    ) -> float:

        return 0.0

    def requires_remeshing(self) -> bool:

        return False

    def project_ncg_search_direction(
        self, search_direction: List[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Parameters
        ----------
        a : list[fenics.Function]
            A function that shall be projected / restricted (will be overwritten)

        Returns
        -------
        None
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
