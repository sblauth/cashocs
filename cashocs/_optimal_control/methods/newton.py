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

"""Truncated Newton methods.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._loggers import debug
from ..._optimal_control import ArmijoLineSearch, ControlOptimizationAlgorithm


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class Newton(ControlOptimizationAlgorithm):
    """A truncated Newton method."""

    def __init__(self, optimization_problem: OptimalControlProblem) -> None:
        """Initializes the Newton method

        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The OptimalControlProblem object
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)
        self.hessian_problem = optimization_problem.hessian_problem

        self.stepsize = 1.0
        self.armijo_stepsize_initial = self.stepsize

        self.has_curvature_info = False

        self.armijo_broken = False

    def run(self) -> None:
        """Performs the optimization via the truncated Newton method

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
                    self.objective_value = self.cost_functional.evaluate()
                    self.converged = True
                    break

            self.relative_norm = self.gradient_norm / self.gradient_norm_initial
            if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
                if self.iteration == 0:
                    self.objective_value = self.cost_functional.evaluate()
                self.converged = True
                break

            self.search_directions = self.hessian_problem.newton_solve()
            self.directional_derivative = self.form_handler.scalar_product(
                self.search_directions, self.gradients
            )
            if self.directional_derivative > 0:
                self.has_curvature_info = False
                debug("Did not compute a descent direction with Newton's method.")
                for i in range(len(self.gradients)):
                    self.search_directions[i].vector().vec().aypx(
                        0.0, -self.gradients[i].vector().vec()
                    )
            else:
                self.has_curvature_info = True

            self.line_search.search(self.search_directions, self.has_curvature_info)

            if self.armijo_broken and self.has_curvature_info:
                for i in range(len(self.gradients)):
                    self.search_directions[i].vector().vec().aypx(
                        0.0, -self.gradients[i].vector().vec()
                    )
                self.has_curvature_info = False
                self.armijo_broken = False
                self.line_search.search(self.search_directions, self.has_curvature_info)

                if self.armijo_broken:
                    self.converged_reason = -2
                    break

            elif self.armijo_broken and not self.has_curvature_info:
                self.converged_reason = -2
                break

            self.iteration += 1
            if self.iteration >= self.maximum_iterations:
                self.converged_reason = -1
                break
