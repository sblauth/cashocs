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

"""Truncated Newton method for PDE constrained optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cashocs._optimization.optimization_algorithms import optimization_algorithm

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import line_search as ls


class NewtonMethod(optimization_algorithm.OptimizationAlgorithm):
    """A truncated Newton method."""

    def __init__(
        self,
        db: database.Database,
        optimization_problem: _typing.OptimizationProblem,
        line_search: ls.LineSearch,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            optimization_problem: The corresponding optimization problem.
            line_search: The corresponding line search.

        """
        super().__init__(db, optimization_problem, line_search)

        self.hessian_problem = optimization_problem.hessian_problem

        self.stepsize = 1.0
        self.armijo_stepsize_initial = self.stepsize

        self.armijo_broken = False

    def run(self) -> None:
        """Solves the optimization problem with the truncated Newton method."""
        while True:
            self.compute_gradient()
            self.gradient_norm = self.compute_gradient_norm()

            if self.convergence_test():
                break

            self.compute_search_direction()
            self.check_for_ascent()

            self.evaluate_cost_functional()

            self.line_search.perform(
                self, self.search_direction, self.has_curvature_info
            )

            if self.line_search_broken and self.has_curvature_info:
                for i in range(len(self.gradient)):
                    self.search_direction[i].vector().vec().aypx(
                        0.0, -self.gradient[i].vector().vec()
                    )
                    self.search_direction[i].vector().apply("")
                self.has_curvature_info = False
                self.line_search_broken = False

                self.line_search.perform(
                    self, self.search_direction, self.has_curvature_info
                )

            self.iteration += 1
            if self.nonconvergence():
                break

    def compute_search_direction(self) -> None:
        """Computes the search direction for the Newton method."""
        self.search_direction = self.hessian_problem.newton_solve()
        self.has_curvature_info = True
