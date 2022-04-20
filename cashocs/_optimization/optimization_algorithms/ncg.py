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

"""Nonlinear CG method for PDE constrained optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cashocs import _utils
from cashocs._optimization.optimization_algorithms import optimization_algorithm

if TYPE_CHECKING:
    from cashocs import types
    from cashocs._optimization import line_search as ls


class NonlinearCGMethod(optimization_algorithm.OptimizationAlgorithm):
    """Nonlinear CG methods for PDE constrained optimization."""

    def __init__(
        self,
        optimization_problem: types.OptimizationProblem,
        line_search: ls.LineSearch,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            line_search: The corresponding line search.

        """
        super().__init__(optimization_problem)
        self.line_search = line_search

        self.gradient_prev = _utils.create_function_list(
            self.form_handler.control_spaces
        )
        self.difference = _utils.create_function_list(self.form_handler.control_spaces)

        self.cg_method = self.config.get("AlgoCG", "cg_method")
        self.cg_periodic_restart = self.config.getboolean(
            "AlgoCG", "cg_periodic_restart"
        )
        self.cg_periodic_its = self.config.getint("AlgoCG", "cg_periodic_its")
        self.cg_relative_restart = self.config.getboolean(
            "AlgoCG", "cg_relative_restart"
        )
        self.cg_restart_tol = self.config.getfloat("AlgoCG", "cg_restart_tol")

        self.memory = 0
        self.beta = 0.0

    def run(self) -> None:
        """Solves the optimization problem with the NCG method."""
        self.initialize_solver()
        self.memory = 0

        while True:
            self.store_previous_gradient()

            self.compute_gradient()
            self.gradient_norm = self.compute_gradient_norm()
            self.compute_beta()

            if self.convergence_test():
                break

            self.compute_search_direction()
            self.restart()
            self.project_ncg_search_direction()
            self.check_for_ascent()

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.perform(
                self, self.search_direction, self.has_curvature_info
            )

            self.iteration += 1
            if self.nonconvergence():
                break

    def _compute_beta_fr(self) -> None:
        """Computes beta for the Fletcher-Reeves method."""
        beta_numerator = self.form_handler.scalar_product(self.gradient, self.gradient)
        beta_denominator = self.form_handler.scalar_product(
            self.gradient_prev, self.gradient_prev
        )
        self.beta = beta_numerator / beta_denominator

    def _compute_beta_pr(self) -> None:
        """Computes beta for the Polak-Ribiere method."""
        self._compute_difference()

        beta_numerator = self.form_handler.scalar_product(
            self.gradient, self.difference
        )
        beta_denominator = self.form_handler.scalar_product(
            self.gradient_prev, self.gradient_prev
        )
        self.beta = beta_numerator / beta_denominator

    def _compute_beta_hs(self) -> None:
        """Computes beta for the Hestenes-Stiefel method."""
        self._compute_difference()

        beta_numerator = self.form_handler.scalar_product(
            self.gradient, self.difference
        )
        beta_denominator = self.form_handler.scalar_product(
            self.difference, self.search_direction
        )
        self.beta = beta_numerator / beta_denominator

    def _compute_beta_dy(self) -> None:
        """Computes beta for the Dai-Yuan method."""
        self._compute_difference()

        beta_numerator = self.form_handler.scalar_product(self.gradient, self.gradient)
        beta_denominator = self.form_handler.scalar_product(
            self.search_direction, self.difference
        )
        self.beta = beta_numerator / beta_denominator

    def _compute_beta_hz(self) -> None:
        """Computes beta for the Hager-Zhang method."""
        self._compute_difference()

        dy = self.form_handler.scalar_product(self.search_direction, self.difference)
        y2 = self.form_handler.scalar_product(self.difference, self.difference)

        for i in range(len(self.gradient)):
            self.difference[i].vector().vec().axpy(
                -2 * y2 / dy, self.search_direction[i].vector().vec()
            )
            self.difference[i].vector().apply("")

        self.beta = (
            self.form_handler.scalar_product(self.difference, self.gradient) / dy
        )

    def compute_beta(self) -> None:
        """Computes the NCG update parameter beta."""
        beta_method = {
            "fr": self._compute_beta_fr,
            "pr": self._compute_beta_pr,
            "hs": self._compute_beta_hs,
            "dy": self._compute_beta_dy,
            "hz": self._compute_beta_hz,
        }

        if self.iteration > 0:
            beta_method[self.cg_method.casefold()]()
        else:
            self.beta = 0.0

    def compute_search_direction(self) -> None:
        """Computes the search direction for the NCG method."""
        for i in range(len(self.gradient)):
            self.search_direction[i].vector().vec().aypx(
                self.beta, -self.gradient[i].vector().vec()
            )
            self.search_direction[i].vector().apply("")

    def restart(self) -> None:
        """Checks, whether the NCG method should be restarted and does the restart."""
        if self.cg_periodic_restart:
            if self.memory < self.cg_periodic_its:
                self.memory += 1
            else:
                for i in range(len(self.gradient)):
                    self.search_direction[i].vector().vec().aypx(
                        0.0, -self.gradient[i].vector().vec()
                    )
                    self.search_direction[i].vector().apply("")
                self.memory = 0
        if self.cg_relative_restart:
            if (
                abs(self.form_handler.scalar_product(self.gradient, self.gradient_prev))
                / pow(self.gradient_norm, 2)
                >= self.cg_restart_tol
            ):
                for i in range(len(self.gradient)):
                    self.search_direction[i].vector().vec().aypx(
                        0.0, -self.gradient[i].vector().vec()
                    )
                    self.search_direction[i].vector().apply("")

    def store_previous_gradient(self) -> None:
        """Stores a copy of the gradient of the previous iteration."""
        for i in range(len(self.gradient)):
            self.gradient_prev[i].vector().vec().aypx(
                0.0, self.gradient[i].vector().vec()
            )
            self.gradient_prev[i].vector().apply("")

    def project_ncg_search_direction(self) -> None:
        """Projects the search direction according to the box constraints."""
        self.optimization_variable_abstractions.project_ncg_search_direction(
            self.search_direction
        )

    def _compute_difference(self) -> None:
        """Computes the difference between current and previous gradients."""
        for i in range(len(self.gradient)):
            self.difference[i].vector().vec().aypx(
                0.0,
                self.gradient[i].vector().vec() - self.gradient_prev[i].vector().vec(),
            )
            self.difference[i].vector().apply("")
