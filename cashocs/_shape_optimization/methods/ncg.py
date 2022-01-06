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

"""Nonlinear conjugate gradient methods for shape optimization.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import numpy as np

from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm
from ..._interfaces.optimization_methods import NCGMixin

if TYPE_CHECKING:
    from ..shape_optimization_problem import ShapeOptimizationProblem


class NCG(NCGMixin, ShapeOptimizationAlgorithm):
    """A nonlinear conjugate gradient (NCG) method for solving shape optimization problems"""

    def __init__(self, optimization_problem: ShapeOptimizationProblem) -> None:
        """A nonlinear cg method to solve the optimization problem

        Additional parameters in the config file:
            cg_method : (one of) FR [Fletcher Reeves], PR [Polak Ribiere],
            HS [Hestenes Stiefel], DY [Dai-Yuan], CD [Conjugate Descent],
            HZ [Hager Zhang]

        Parameters
        ----------
        optimization_problem : ShapeOptimizationProblem
            the OptimalControlProblem object
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)

    def run(self) -> None:
        """Performs the optimization via the nonlinear cg method

        Returns
        -------
        None
        """

        self.initialize_solver()
        self.memory = 0

        while True:

            self.store_previous_gradient()

            self.compute_gradient()
            self.gradient_norm = np.sqrt(self.gradient_problem.gradient_norm_squared)
            self.compute_beta()

            if self.convergence_test():
                break

            self.compute_search_direction()
            self.restart()
            self.check_for_ascent()

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
