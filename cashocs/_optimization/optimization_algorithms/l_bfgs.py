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

"""Limited Memory BFGS method for PDE constrained optimization."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, List

import fenics
import numpy as np

from cashocs._optimization.optimization_algorithms import optimization_algorithm

if TYPE_CHECKING:
    from cashocs._optimization import optimization_problem as op
    from cashocs._optimization import line_search as ls


class LBFGSMethod(optimization_algorithm.OptimizationAlgorithm):
    """A limited memory BFGS method."""

    def __init__(
        self, optimization_problem: op.OptimizationProblem, line_search: ls.LineSearch
    ) -> None:
        """
        Args:
            optimization_problem: The corresponding optimization problem.
            line_search: The corresponding line search.
        """

        super().__init__(optimization_problem)
        self.line_search = line_search

        self.temp = [fenics.Function(V) for V in self.form_handler.control_spaces]

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
            self.history_alpha = deque()
            self.gradient_prev = [
                fenics.Function(V) for V in self.form_handler.control_spaces
            ]
            self.y_k = [fenics.Function(V) for V in self.form_handler.control_spaces]
            self.s_k = [fenics.Function(V) for V in self.form_handler.control_spaces]

    def run(self) -> None:
        """Solves the optimization problem with the L-BFGS method."""

        self.initialize_solver()
        self.compute_gradient()
        self.form_handler.compute_active_sets()
        self.gradient_norm = (
            self.optimization_variable_abstractions.compute_gradient_norm()
        )

        self.converged = self.convergence_test()

        while not self.converged:
            self.compute_search_direction(self.gradient)
            self.check_for_ascent()

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.perform(
                self, self.search_direction, self.has_curvature_info
            )

            self.iteration += 1
            if self.nonconvergence():
                break

            self.store_previous_gradient()
            self.compute_gradient()
            self.form_handler.compute_active_sets()
            self.gradient_norm = (
                self.optimization_variable_abstractions.compute_gradient_norm()
            )
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial

            if self.convergence_test():
                break

            self.update_hessian_approximation()

    def _first_loop(self) -> None:

        for i, _ in enumerate(self.history_s):
            alpha = self.history_rho[i] * self.form_handler.scalar_product(
                self.history_s[i], self.search_direction
            )
            self.history_alpha.append(alpha)
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().axpy(
                    -alpha, self.history_y[i][j].vector().vec()
                )

    def _second_loop(self) -> None:

        for i, _ in enumerate(self.history_s):
            beta = self.history_rho[-1 - i] * self.form_handler.scalar_product(
                self.history_y[-1 - i], self.search_direction
            )

            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().axpy(
                    self.history_alpha[-1 - i] - beta,
                    self.history_s[-1 - i][j].vector().vec(),
                )

    def _bfgs_scaling(self) -> None:

        if self.use_bfgs_scaling and self.iteration > 0:
            factor = self.form_handler.scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self.form_handler.scalar_product(self.history_y[0], self.history_y[0])
        else:
            factor = 1.0

        for j in range(len(self.gradient)):
            self.search_direction[j].vector().vec().scale(factor)

    def compute_search_direction(self, grad: List[fenics.Function]) -> None:
        """Computes the search direction for the BFGS method with a double loop.

        Args:
            grad: The current gradient

        Returns:
            A function corresponding to the current / next search direction
        """

        if self.bfgs_memory_size > 0 and len(self.history_s) > 0:
            self.history_alpha.clear()
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().aypx(
                    0.0, grad[j].vector().vec()
                )

            self.form_handler.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )

            self._first_loop()
            self._bfgs_scaling()

            self.form_handler.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )

            self._second_loop()

            self.form_handler.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )
            self.form_handler.restrict_to_active_set(self.gradient, self.temp)
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().axpy(
                    1.0, self.temp[j].vector().vec()
                )
                self.search_direction[j].vector().vec().scale(-1.0)

        else:
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().aypx(
                    0.0, -grad[j].vector().vec()
                )

    def store_previous_gradient(self) -> None:
        """Stores a copy of the gradient in the previous iteration."""

        if self.bfgs_memory_size > 0:
            for i in range(len(self.gradient)):
                self.gradient_prev[i].vector().vec().aypx(
                    0.0, self.gradient[i].vector().vec()
                )

    def update_hessian_approximation(self) -> None:
        """Updates the approximation of the inverse Hessian."""

        if self.bfgs_memory_size > 0:
            for i in range(len(self.gradient)):
                self.y_k[i].vector().vec().aypx(
                    0.0,
                    self.gradient[i].vector().vec()
                    - self.gradient_prev[i].vector().vec(),
                )
                self.s_k[i].vector().vec().aypx(
                    0.0,
                    self.stepsize * self.search_direction[i].vector().vec(),
                )

            self.form_handler.restrict_to_inactive_set(self.y_k, self.y_k)
            self.form_handler.restrict_to_inactive_set(self.s_k, self.s_k)

            self.history_y.appendleft([x.copy(True) for x in self.y_k])
            self.history_s.appendleft([x.copy(True) for x in self.s_k])
            curvature_condition = self.form_handler.scalar_product(self.y_k, self.s_k)

            if (
                curvature_condition
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
                rho = 1 / curvature_condition
                self.history_rho.appendleft(rho)

            if len(self.history_s) > self.bfgs_memory_size:
                self.history_s.pop()
                self.history_y.pop()
                self.history_rho.pop()
