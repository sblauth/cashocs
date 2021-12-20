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

"""Gradient descent methods for shape optimization.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm


if TYPE_CHECKING:
    from ..shape_optimization_problem import ShapeOptimizationProblem


class GradientDescent(ShapeOptimizationAlgorithm):
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
        self.has_curvature_info = False

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

        try:
            self.iteration = self.temp_dict["OptimizationRoutine"].get(
                "iteration_counter", 0
            )
            self.gradient_norm_initial = self.temp_dict["OptimizationRoutine"].get(
                "gradient_norm_initial", 0.0
            )
        except TypeError:
            self.iteration = 0
            self.gradient_norm_initial = 0.0

        self.relative_norm = 1.0
        self.state_problem.has_solution = False

        while True:

            self.adjoint_problem.has_solution = False
            self.shape_gradient_problem.has_solution = False
            self.shape_gradient_problem.solve()
            self.gradient_norm = np.sqrt(
                self.shape_gradient_problem.gradient_norm_squared
            )

            if self.iteration == 0:
                self.gradient_norm_initial = self.gradient_norm
                if self.gradient_norm_initial == 0:
                    self.converged = True
                    break

            self.relative_norm = self.gradient_norm / self.gradient_norm_initial
            if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
                self.converged = True
                break

            self.search_direction.vector().vec().aypx(
                0.0, -self.gradient.vector().vec()
            )

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
