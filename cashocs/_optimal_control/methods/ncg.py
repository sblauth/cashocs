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

"""Nonlinear conjugate gradient methods.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import fenics
import numpy as np

from ..._interfaces.optimization_methods import NCGMixin
from ..._optimal_control import ArmijoLineSearch, ControlOptimizationAlgorithm


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class NCG(NCGMixin, ControlOptimizationAlgorithm):
    """Nonlinear conjugate gradient method."""

    def __init__(self, optimization_problem: OptimalControlProblem):
        """
        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The OptimalControlProblem object
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)
        self.control_constraints = optimization_problem.control_constraints

    def project_direction(self, a: List[fenics.Function]) -> None:
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
                        a[j].vector()[:] < 0.0,
                    ),
                    np.logical_and(
                        self.controls[j].vector()[:]
                        >= self.control_constraints[j][1].vector()[:],
                        a[j].vector()[:] > 0.0,
                    ),
                )
            ).nonzero()[0]

            a[j].vector()[idx] = 0.0

    def run(self) -> None:
        """Performs the optimization via the nonlinear cg method

        Returns
        -------
        None
        """

        self.initialize_solver()
        self.memory = 0

        while True:

            for i in range(self.form_handler.control_dim):
                self.gradient_prev[i].vector().vec().aypx(
                    0.0, self.gradient[i].vector().vec()
                )

            self.compute_gradient()

            self.gradient_norm = np.sqrt(self._stationary_measure_squared())
            self.compute_beta()

            if self.convergence_test():
                break

            self.compute_search_direction()
            self.restart()
            self.project_direction(self.search_direction)
            self.check_for_ascent()

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
