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

"""Gradient descent methods.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._interfaces.optimization_methods import GradientDescentMixin

from ..._optimal_control import ArmijoLineSearch, ControlOptimizationAlgorithm


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class GradientDescent(GradientDescentMixin, ControlOptimizationAlgorithm):
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

        self.initialize_solver()

        while True:

            self.compute_gradient()
            self.gradient_norm = np.sqrt(self._stationary_measure_squared())

            if self.convergence_test():
                break

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.compute_search_direction()
            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
