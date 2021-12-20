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

"""Limited memory BFGS methods.

"""

from __future__ import annotations

from _collections import deque
from typing import TYPE_CHECKING, List

import fenics
import numpy as np

from ..._loggers import debug
from ..._optimal_control import ArmijoLineSearch, ControlOptimizationAlgorithm


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class LBFGS(ControlOptimizationAlgorithm):
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

        self.temp = [fenics.Function(V) for V in optimization_problem.control_spaces]
        self.storage_y = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]
        self.storage_s = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]

        self.bfgs_memory_size = self.config.getint(
            "AlgoLBFGS", "bfgs_memory_size", fallback=5
        )
        self.use_bfgs_scaling = self.config.getboolean(
            "AlgoLBFGS", "use_bfgs_scaling", fallback=True
        )

        self.has_curvature_info = False

        if self.bfgs_memory_size > 0:
            self.history_s = deque()
            self.history_y = deque()
            self.history_rho = deque()
            self.gradients_prev = [
                fenics.Function(V) for V in optimization_problem.control_spaces
            ]
            self.y_k = [fenics.Function(V) for V in optimization_problem.control_spaces]
            self.s_k = [fenics.Function(V) for V in optimization_problem.control_spaces]

    def compute_search_direction(
        self, grad: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Computes the search direction for the BFGS method with a double loop

        Parameters
        ----------
        grad : list[fenics.Function]
            The current gradient

        Returns
        -------
        search_directions : list[fenics.Function]
            A function corresponding to the current / next search direction
        """

        if self.bfgs_memory_size > 0 and len(self.history_s) > 0:
            history_alpha = deque()
            for j in range(len(self.controls)):
                self.search_directions[j].vector().vec().aypx(
                    0.0, grad[j].vector().vec()
                )

            self.form_handler.restrict_to_inactive_set(
                self.search_directions, self.search_directions
            )

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self.form_handler.scalar_product(
                    self.history_s[i], self.search_directions
                )
                history_alpha.append(alpha)
                for j in range(len(self.controls)):
                    self.search_directions[j].vector().vec().axpy(
                        -alpha, self.history_y[i][j].vector().vec()
                    )

            if self.use_bfgs_scaling and self.iteration > 0:
                factor = self.form_handler.scalar_product(
                    self.history_y[0], self.history_s[0]
                ) / self.form_handler.scalar_product(
                    self.history_y[0], self.history_y[0]
                )
            else:
                factor = 1.0

            for j in range(len(self.controls)):
                self.search_directions[j].vector().vec().scale(factor)

            self.form_handler.restrict_to_inactive_set(
                self.search_directions, self.search_directions
            )

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self.form_handler.scalar_product(
                    self.history_y[-1 - i], self.search_directions
                )

                for j in range(len(self.controls)):
                    self.search_directions[j].vector().vec().axpy(
                        history_alpha[-1 - i] - beta,
                        self.history_s[-1 - i][j].vector().vec(),
                    )

            self.form_handler.restrict_to_inactive_set(
                self.search_directions, self.search_directions
            )
            self.form_handler.restrict_to_active_set(self.gradients, self.temp)
            for j in range(len(self.controls)):
                self.search_directions[j].vector().vec().axpy(
                    1.0, self.temp[j].vector().vec()
                )
                self.search_directions[j].vector().vec().scale(-1.0)

        else:
            for j in range(len(self.controls)):
                self.search_directions[j].vector().vec().aypx(
                    0.0, -grad[j].vector().vec()
                )

        return self.search_directions

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
            self.search_directions = self.compute_search_direction(self.gradients)

            self.directional_derivative = self.form_handler.scalar_product(
                self.search_directions, self.gradients
            )
            if self.directional_derivative > 0:
                debug("No descent direction found with L-BFGS")
                for j in range(self.form_handler.control_dim):
                    self.search_directions[j].vector().vec().aypx(
                        0.0, -self.gradients[j].vector().vec()
                    )

            self.line_search.search(self.search_directions, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break

            if self.bfgs_memory_size > 0:
                for i in range(len(self.gradients)):
                    self.gradients_prev[i].vector().vec().aypx(
                        0.0, self.gradients[i].vector().vec()
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
                for i in range(len(self.gradients)):
                    self.storage_y[i].vector().vec().aypx(
                        0.0,
                        self.gradients[i].vector().vec()
                        - self.gradients_prev[i].vector().vec(),
                    )
                    self.storage_s[i].vector().vec().aypx(
                        0.0,
                        self.stepsize * self.search_directions[i].vector().vec(),
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
