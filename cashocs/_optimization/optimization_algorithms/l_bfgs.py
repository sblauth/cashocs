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

"""Limited Memory BFGS method for PDE constrained optimization."""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import fenics

from cashocs import _utils
from cashocs._optimization.optimization_algorithms import optimization_algorithm

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import line_search as ls


class LBFGSMethod(optimization_algorithm.OptimizationAlgorithm):
    """A limited memory BFGS method."""

    history_s: collections.deque
    history_y: collections.deque
    history_rho: collections.deque
    history_alpha: collections.deque

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

        self.bfgs_memory_size = self.config.getint("AlgoLBFGS", "bfgs_memory_size")
        self.use_bfgs_scaling = self.config.getboolean("AlgoLBFGS", "use_bfgs_scaling")
        self.bfgs_periodic_restart = self.config.getint(
            "AlgoLBFGS", "bfgs_periodic_restart"
        )
        self.periodic_its = 0
        self.damped = self.config.getboolean("AlgoLBFGS", "damped")

        self._init_helpers()

    def _init_helpers(self) -> None:
        """Initializes the helper functions."""
        self.temp = _utils.create_function_list(self.db.function_db.control_spaces)
        if self.bfgs_memory_size > 0:
            self.history_s = collections.deque()
            self.history_y = collections.deque()
            self.history_rho = collections.deque()
            self.history_alpha = collections.deque()
            self.gradient_prev = _utils.create_function_list(
                self.db.function_db.control_spaces
            )
            self.y_k = _utils.create_function_list(self.db.function_db.control_spaces)
            self.s_k = _utils.create_function_list(self.db.function_db.control_spaces)
            self.r_k = _utils.create_function_list(self.db.function_db.control_spaces)
            self.storage = _utils.create_function_list(
                self.db.function_db.control_spaces
            )
            self.approx_hessian_applied_to_y = _utils.create_function_list(
                self.db.function_db.control_spaces
            )

    def run(self) -> None:
        """Solves the optimization problem with the L-BFGS method."""
        self.periodic_its = 0
        self.compute_gradient()
        self.optimization_variable_abstractions.compute_active_sets()
        self.gradient_norm = (
            self.optimization_variable_abstractions.compute_gradient_norm()
        )

        self.converged = self.convergence_test()

        while not self.converged:
            self.compute_search_direction(self.gradient)
            self.check_restart()
            self.check_for_ascent()

            self.evaluate_cost_functional()

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

            self.store_previous_gradient()
            self.compute_gradient()
            self.optimization_variable_abstractions.compute_active_sets()
            self.gradient_norm = (
                self.optimization_variable_abstractions.compute_gradient_norm()
            )
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial

            if self.convergence_test():
                break

            self.update_hessian_approximation()

    def _first_loop(self) -> None:
        """Performs the first of the two L-BFGS loops."""
        for i, _ in enumerate(self.history_s):
            alpha = self.history_rho[i] * self.form_handler.scalar_product(
                self.history_s[i], self.search_direction
            )
            self.history_alpha.append(alpha)
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().axpy(
                    -alpha, self.history_y[i][j].vector().vec()
                )
                self.search_direction[j].vector().apply("")

    def _second_loop(self) -> None:
        """Performs the second of the two L-BFGS loops."""
        for i, _ in enumerate(self.history_s):
            beta = self.history_rho[-1 - i] * self.form_handler.scalar_product(
                self.history_y[-1 - i], self.search_direction
            )

            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().axpy(
                    self.history_alpha[-1 - i] - beta,
                    self.history_s[-1 - i][j].vector().vec(),
                )
                self.search_direction[j].vector().apply("")

    def _bfgs_scaling(self) -> None:
        """Scales the BFGS search direction."""
        if self.use_bfgs_scaling and self.iteration > 0:
            factor = self.form_handler.scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self.form_handler.scalar_product(self.history_y[0], self.history_y[0])
        else:
            factor = 1.0

        for j in range(len(self.gradient)):
            self.search_direction[j].vector().vec().scale(factor)
            self.search_direction[j].vector().apply("")

    def compute_search_direction(self, grad: list[fenics.Function]) -> None:
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
                self.search_direction[j].vector().apply("")

            self.optimization_variable_abstractions.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )

            self._first_loop()
            self._bfgs_scaling()

            self.optimization_variable_abstractions.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )

            self._second_loop()

            self.optimization_variable_abstractions.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )
            self.optimization_variable_abstractions.restrict_to_active_set(
                self.gradient, self.temp
            )
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().axpy(
                    1.0, self.temp[j].vector().vec()
                )
                self.search_direction[j].vector().apply("")
                self.search_direction[j].vector().vec().scale(-1.0)
                self.search_direction[j].vector().apply("")

        else:
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().aypx(
                    0.0, -grad[j].vector().vec()
                )
                self.search_direction[j].vector().apply("")

    def store_previous_gradient(self) -> None:
        """Stores a copy of the gradient in the previous iteration."""
        if self.bfgs_memory_size > 0:
            for i in range(len(self.gradient)):
                self.gradient_prev[i].vector().vec().aypx(
                    0.0, self.gradient[i].vector().vec()
                )
                self.gradient_prev[i].vector().apply("")

    def update_hessian_approximation(self) -> None:
        """Updates the approximation of the inverse Hessian."""
        if self.bfgs_memory_size > 0:
            for i in range(len(self.gradient)):
                self.y_k[i].vector().vec().aypx(
                    0.0,
                    self.gradient[i].vector().vec()
                    - self.gradient_prev[i].vector().vec(),
                )
                self.y_k[i].vector().apply("")
                self.s_k[i].vector().vec().axpby(
                    self.stepsize,
                    0.0,
                    self.search_direction[i].vector().vec(),
                )
                self.s_k[i].vector().apply("")

            self.optimization_variable_abstractions.restrict_to_inactive_set(
                self.y_k, self.y_k
            )
            self.optimization_variable_abstractions.restrict_to_inactive_set(
                self.s_k, self.s_k
            )

            curvature_condition = self.form_handler.scalar_product(self.y_k, self.s_k)

            self._compute_approximate_inverse_hessian_application(
                self.y_k, self.approx_hessian_applied_to_y
            )
            gamma = self.form_handler.scalar_product(
                self.y_k, self.approx_hessian_applied_to_y
            )

            if not self.damped:
                self.history_y.appendleft([x.copy(True) for x in self.y_k])
                self.history_s.appendleft([x.copy(True) for x in self.s_k])
                if curvature_condition <= 0.0:
                    self.has_curvature_info = False
                    self.history_s.clear()
                    self.history_y.clear()
                    self.history_rho.clear()

                else:
                    self.has_curvature_info = True
                    rho = 1 / curvature_condition
                    self.history_rho.appendleft(rho)

            else:
                if curvature_condition >= 0.2 * gamma:
                    phi = 1.0
                else:
                    phi = (0.8 * gamma) / (gamma - curvature_condition)

                for i in range(len(self.gradient)):
                    self.r_k[i].vector().vec().axpby(
                        1.0, 0.0, self.approx_hessian_applied_to_y[i].vector().vec()
                    )
                    self.r_k[i].vector().apply("")
                    self.r_k[i].vector().vec().axpby(
                        phi, 1.0 - phi, self.s_k[i].vector().vec()
                    )
                    self.r_k[i].vector().apply("")

                self.history_s.appendleft([x.copy(True) for x in self.r_k])
                self.history_y.appendleft([x.copy(True) for x in self.y_k])
                self.has_curvature_info = True
                rho = 1 / self.form_handler.scalar_product(self.y_k, self.r_k)
                self.history_rho.appendleft(rho)

            if len(self.history_s) > self.bfgs_memory_size:
                self.history_s.pop()
                self.history_y.pop()
                self.history_rho.pop()

    def check_restart(self) -> None:
        """Checks, whether a restart should be performed and does so, if necessary."""
        if self.bfgs_periodic_restart > 0:
            if self.periodic_its < self.bfgs_periodic_restart:
                self.periodic_its += 1
                self.is_restarted = False
            else:
                for i in range(len(self.gradient)):
                    self.search_direction[i].vector().vec().aypx(
                        0.0, -self.gradient[i].vector().vec()
                    )
                    self.search_direction[i].vector().apply("")
                self.periodic_its = 0
                self.has_curvature_info = False
                self.is_restarted = True

                self.history_s.clear()
                self.history_y.clear()
                self.history_rho.clear()

    def _compute_approximate_inverse_hessian_application(
        self, y: list[fenics.Function], res: list[fenics.Function]
    ) -> None:
        """Computes the application of the approximate inverse Hessian to y.

        Args:
            y: The vector, on which the approximate inverse Hessian is applied to.
            res: The result of the application, i.e., Hy.

        """
        for i in range(len(self.gradient)):
            self.storage[i].vector().vec().aypx(
                0.0, self.search_direction[i].vector().vec()
            )
            self.storage[i].vector().apply("")

        self.compute_search_direction(y)
        for i in range(len(self.gradient)):
            res[i].vector().vec().aypx(0.0, -self.search_direction[i].vector().vec())
            res[i].vector().apply("")
            self.search_direction[i].vector().vec().aypx(
                0.0, self.storage[i].vector().vec()
            )
            self.search_direction[i].vector().apply("")
