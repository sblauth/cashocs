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

"""Limited memory BFGS methods.

"""

from __future__ import annotations

from _collections import deque
from typing import TYPE_CHECKING, List

import fenics
import numpy as np

from ..._loggers import debug
from ..._optimal_control import ArmijoLineSearch, ControlOptimizationAlgorithm
from ..._interfaces.optimization_methods import LBFGSMixin


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class LBFGS(LBFGSMixin, ControlOptimizationAlgorithm):
    """A limited memory BFGS method"""

    def __init__(self, optimization_problem: OptimalControlProblem) -> None:
        """Initializes the L-BFGS method.

        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The optimization problem to be solved
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)

    def run(self) -> None:
        """Performs the optimization via the limited memory BFGS method

        Returns
        -------
        None
        """

        self.initialize_solver()
        self.compute_gradient()
        self.form_handler.compute_active_sets()
        self.gradient_norm = np.sqrt(self._stationary_measure_squared())

        self.converged = self.convergence_test()

        while not self.converged:
            self.compute_search_direction(self.gradient)
            self.check_for_ascent()

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break

            self.store_previous_gradient()

            self.compute_gradient()
            self.form_handler.compute_active_sets()
            self.gradient_norm = np.sqrt(self._stationary_measure_squared())
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial

            if self.convergence_test():
                break

            self.update_hessian_approximation()
