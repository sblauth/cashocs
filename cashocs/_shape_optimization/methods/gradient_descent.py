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

"""Gradient descent methods for shape optimization.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm
from ..._interfaces.optimization_methods import GradientDescentMixin

if TYPE_CHECKING:
    from ..shape_optimization_problem import ShapeOptimizationProblem


class GradientDescent(GradientDescentMixin, ShapeOptimizationAlgorithm):
    """A gradient descent method for shape optimization"""

    def __init__(self, optimization_problem: ShapeOptimizationProblem) -> None:
        """A gradient descent method to solve the shape optimization problem

        Parameters
        ----------
        optimization_problem : ShapeOptimizationProblem
            The shape optimization problem to be solved
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)

    def run(self) -> None:
        """Performs the optimization via the gradient descent method

        Notes
        -----
        The result of the optimization is found in the user-defined inputs for the
        optimization problem.

        Returns
        -------
        None
        """

        self.initialize_solver()

        while True:

            self.compute_gradient()
            self.gradient_norm = np.sqrt(self.gradient_problem.gradient_norm_squared)

            if self.convergence_test():
                break

            self.objective_value = self.cost_functional.evaluate()
            self.output()
            
            self.compute_search_direction()
            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
