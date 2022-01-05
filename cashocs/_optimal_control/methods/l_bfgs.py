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

        self.storage_y = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]
        self.storage_s = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]

    def run(self) -> None:
        """Performs the optimization via the limited memory BFGS method

        Returns
        -------
        None
        """

        self.converged = False

        self.iteration = 0
        self.relative_norm = 1.0
        self.state_problem.has_solution = False

        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False
        self.gradient_problem.solve()
        self.gradient_norm = np.sqrt(self._stationary_measure_squared())
        self.gradient_norm_initial = self.gradient_norm
        if self.gradient_norm_initial == 0:
            self.converged = True
        self.form_handler.compute_active_sets()

        while not self.converged:
            self.search_direction = self.compute_search_direction(self.gradient)

            self.directional_derivative = self.form_handler.scalar_product(
                self.search_direction, self.gradient
            )
            if self.directional_derivative > 0:
                debug("No descent direction found with L-BFGS")
                for j in range(self.form_handler.control_dim):
                    self.search_direction[j].vector().vec().aypx(
                        0.0, -self.gradient[j].vector().vec()
                    )

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break

            if self.bfgs_memory_size > 0:
                for i in range(len(self.gradient)):
                    self.gradients_prev[i].vector().vec().aypx(
                        0.0, self.gradient[i].vector().vec()
                    )

            self.adjoint_problem.has_solution = False
            self.gradient_problem.has_solution = False
            self.gradient_problem.solve()
            self.form_handler.compute_active_sets()

            self.gradient_norm = np.sqrt(self._stationary_measure_squared())
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial

            if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
                self.converged = True
                break

            if self.bfgs_memory_size > 0:
                for i in range(len(self.gradient)):
                    self.storage_y[i].vector().vec().aypx(
                        0.0,
                        self.gradient[i].vector().vec()
                        - self.gradients_prev[i].vector().vec(),
                    )
                    self.storage_s[i].vector().vec().aypx(
                        0.0,
                        self.stepsize * self.search_direction[i].vector().vec(),
                    )

                self.form_handler.restrict_to_inactive_set(self.storage_y, self.y_k)
                self.form_handler.restrict_to_inactive_set(self.storage_s, self.s_k)

                self.history_y.appendleft([x.copy(True) for x in self.y_k])
                self.history_s.appendleft([x.copy(True) for x in self.s_k])
                self.curvature_condition = self.form_handler.scalar_product(
                    self.y_k, self.s_k
                )

                if (
                    self.curvature_condition
                    / np.sqrt(
                        self.form_handler.scalar_product(self.s_k, self.s_k)
                        * self.form_handler.scalar_product(self.y_k, self.y_k)
                    )
                    <= 1e-14
                ):
                    self.has_curvature_info = False

                    self.history_s.clear()
                    self.history_y.clear()
                    self.history_rho.clear()

                else:
                    self.has_curvature_info = True
                    rho = 1 / self.curvature_condition
                    self.history_rho.appendleft(rho)

                if len(self.history_s) > self.bfgs_memory_size:
                    self.history_s.pop()
                    self.history_y.pop()
                    self.history_rho.pop()
