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

from __future__ import annotations

from _collections import deque

from typing import TYPE_CHECKING, List

import fenics

if TYPE_CHECKING:
    from ..optimization_problem import OptimizationProblem


class LBFGSMixin:
    def __init__(self, optimization_problem: OptimizationProblem) -> None:

        super().__init__(optimization_problem)
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
            self.gradients_prev = [
                fenics.Function(V) for V in self.form_handler.control_spaces
            ]
            self.y_k = [fenics.Function(V) for V in self.form_handler.control_spaces]
            self.s_k = [fenics.Function(V) for V in self.form_handler.control_spaces]

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
        search_direction : list[fenics.Function]
            A function corresponding to the current / next search direction
        """

        if self.bfgs_memory_size > 0 and len(self.history_s) > 0:
            history_alpha = deque()
            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().aypx(
                    0.0, grad[j].vector().vec()
                )

            self.form_handler.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self.form_handler.scalar_product(
                    self.history_s[i], self.search_direction
                )
                history_alpha.append(alpha)
                for j in range(len(self.gradient)):
                    self.search_direction[j].vector().vec().axpy(
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

            for j in range(len(self.gradient)):
                self.search_direction[j].vector().vec().scale(factor)

            self.form_handler.restrict_to_inactive_set(
                self.search_direction, self.search_direction
            )

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self.form_handler.scalar_product(
                    self.history_y[-1 - i], self.search_direction
                )

                for j in range(len(self.gradient)):
                    self.search_direction[j].vector().vec().axpy(
                        history_alpha[-1 - i] - beta,
                        self.history_s[-1 - i][j].vector().vec(),
                    )

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

        return self.search_direction
