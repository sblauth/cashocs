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

"""Limited memory BFGS methods for shape optimization.

"""

from __future__ import annotations

from _collections import deque
from typing import TYPE_CHECKING, List

import fenics
import numpy as np

from ..._loggers import debug
from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm
from ..._interfaces.optimization_methods import LBFGSMixin

if TYPE_CHECKING:
    from ..shape_optimization_problem import ShapeOptimizationProblem


class LBFGS(LBFGSMixin, ShapeOptimizationAlgorithm):
    """A limited memory BFGS (L-BFGS) method for solving shape optimization problems"""

    def __init__(self, optimization_problem: ShapeOptimizationProblem) -> None:
        """Implements the L-BFGS method for solving the optimization problem

        Parameters
        ----------
        optimization_problem : ShapeOptimizationProblem
            The shape optimization problem
        """

        super().__init__(optimization_problem)

        self.line_search = ArmijoLineSearch(self)

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
            self.gradient_prev = fenics.Function(self.form_handler.deformation_space)
            self.y_k = [fenics.Function(self.form_handler.deformation_space)]
            self.s_k = [fenics.Function(self.form_handler.deformation_space)]

    def run(self) -> None:
        """Performs the optimization via the limited memory BFGS method

        Returns
        -------
        None
        """

        self.converged = False

        try:
            self.iteration = self.temp_dict["OptimizationRoutine"].get(
                "iteration_counter", 0
            )
            self.gradient_norm_initial = self.temp_dict["OptimizationRoutine"].get(
                "gradient_norm_initial", 0.0
            )
        except TypeError:
            self.iteration = 0
            self.gradient_norm_initial = 0.0
            self.relative_norm = 1.0
        self.state_problem.has_solution = False

        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False
        self.gradient_problem.solve()
        self.gradient_norm = np.sqrt(self.gradient_problem.gradient_norm_squared)

        if self.iteration == 0:
            self.gradient_norm_initial = self.gradient_norm

        if self.gradient_norm_initial == 0.0:
            self.converged = True
        else:
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial

        if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
            self.converged = True

        self.objective_value = self.cost_functional.evaluate()

        while not self.converged:
            self.search_direction = self.compute_search_direction(self.gradient)

            self.directional_derivative = self.form_handler.scalar_product(
                self.gradient, self.search_direction
            )
            if self.directional_derivative > 0:
                debug("No descent direction found with L-BFGS")
                self.search_direction[0].vector().vec().aypx(
                    0.0, -self.gradient[0].vector().vec()
                )

            self.objective_value = self.cost_functional.evaluate()
            self.output()

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break

            if self.bfgs_memory_size > 0:
                self.gradient_prev.vector().vec().aypx(
                    0.0, self.gradient[0].vector().vec()
                )

            self.adjoint_problem.has_solution = False
            self.gradient_problem.has_solution = False
            self.gradient_problem.solve()

            self.gradient_norm = np.sqrt(self.gradient_problem.gradient_norm_squared)
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial

            if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
                self.converged = True
                break

            if self.bfgs_memory_size > 0:
                self.y_k[0].vector().vec().aypx(
                    0.0,
                    self.gradient[0].vector().vec() - self.gradient_prev.vector().vec(),
                )
                self.s_k[0].vector().vec().aypx(
                    0.0, self.stepsize * self.search_direction[0].vector().vec()
                )

                self.history_y.appendleft([self.y_k[0].copy(True)])
                self.history_s.appendleft([self.s_k[0].copy(True)])

                self.curvature_condition = self.form_handler.scalar_product(
                    self.y_k, self.s_k
                )

                if (
                    self.curvature_condition
                    / np.sqrt(
                        self.form_handler.scalar_product(self.y_k, self.y_k)
                        * self.form_handler.scalar_product(self.s_k, self.s_k)
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
