# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Gradient descent methods.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._optimal_control import ArmijoLineSearch, ControlOptimizationAlgorithm


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class GradientDescent(ControlOptimizationAlgorithm):
    """A gradient descent method"""

    def __init__(self, optimization_problem: OptimalControlProblem) -> None:
        """
        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The OptimalControlProblem object
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)

    def run(self) -> None:
        """Performs the optimization via the gradient descent method

        Returns
        -------
        None
        """

        self.iteration = 0
        self.relative_norm = 1.0
        self.state_problem.has_solution = False

        while True:

            self.adjoint_problem.has_solution = False
            self.gradient_problem.has_solution = False
            self.gradient_problem.solve()
            self.gradient_norm = np.sqrt(self._stationary_measure_squared())

            if self.iteration == 0:
                self.gradient_norm_initial = self.gradient_norm
                if self.gradient_norm_initial == 0:
                    self.converged = True
                    break

            self.relative_norm = self.gradient_norm / self.gradient_norm_initial
            if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
                if self.iteration == 0:
                    self.objective_value = self.cost_functional.evaluate()
                self.converged = True
                break

            for i in range(len(self.controls)):
                self.search_directions[i].vector().vec().aypx(
                    0.0, -self.gradients[i].vector().vec()
                )

            self.line_search.search(self.search_directions, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
