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

"""Gradient descent method for PDE constrained optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cashocs._optimization.optimization_algorithms import optimization_algorithm

if TYPE_CHECKING:
    from cashocs import types
    from cashocs._optimization import line_search as ls


class GradientDescentMethod(optimization_algorithm.OptimizationAlgorithm):
    """A gradient descent method."""

    def __init__(
        self,
        optimization_problem: types.OptimizationProblem,
        line_search: ls.LineSearch,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            line_search: The corresponding line search.

        """
        super().__init__(optimization_problem)
        self.line_search = line_search

    def run(self) -> None:
        """Performs the optimization with the gradient descent method."""
        self.initialize_solver()

        while True:

            self.compute_gradient()
            self.gradient_norm = self.compute_gradient_norm()

            if self.convergence_test():
                break

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.compute_search_direction()
            self.line_search.perform(
                self, self.search_direction, self.has_curvature_info
            )

            self.iteration += 1
            if self.nonconvergence():
                break

    def compute_search_direction(self) -> None:
        """Computes the search direction for the gradient descent method."""
        for i in range(len(self.gradient)):
            self.search_direction[i].vector().vec().aypx(
                0.0, -self.gradient[i].vector().vec()
            )
            self.search_direction[i].vector().apply("")
