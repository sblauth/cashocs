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

"""Gradient descent method for PDE constrained optimization."""

from __future__ import annotations

from cashocs._optimization.optimization_algorithms import optimization_algorithm


class GradientDescentMethod(optimization_algorithm.OptimizationAlgorithm):
    """A gradient descent method."""

    def run(self) -> None:
        """Performs the optimization with the gradient descent method."""
        while True:
            self.compute_gradient()
            self.gradient_norm = self.compute_gradient_norm()

            if self.convergence_test():
                break

            self.evaluate_cost_functional()

            self.compute_search_direction()
            self.project_search_direction()
            self.line_search.perform(
                self,
                self.search_direction,
                self.has_curvature_info,
                self.active_idx,
                self.constraint_gradient,
                self.dropped_idx,
            )

            self.iteration += 1
            if self.nonconvergence():
                break

    def compute_search_direction(self) -> None:
        """Computes the search direction for the (projected) gradient descent method."""
        for i in range(len(self.db.function_db.gradient)):
            self.search_direction[i].vector().vec().aypx(
                0.0, -self.db.function_db.gradient[i].vector().vec()
            )
            self.search_direction[i].vector().apply("")
