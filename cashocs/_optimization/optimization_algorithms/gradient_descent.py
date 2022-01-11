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

"""Gradient descent method for PDE constrained optimization.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .optimization_algorithm import OptimizationAlgorithm



if TYPE_CHECKING:
    from ..line_search import LineSearch
    from ..optimization_problem import OptimizationProblem


class GradientDescentMethod(OptimizationAlgorithm):
    def __init__(
        self, optimization_problem: OptimizationProblem, line_search: LineSearch
    ) -> None:

        super().__init__(optimization_problem)
        self.line_search = line_search

    def run(self) -> None:

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
        for i in range(len(self.gradient)):
            self.search_direction[i].vector().vec().aypx(
                0.0, -self.gradient[i].vector().vec()
            )
