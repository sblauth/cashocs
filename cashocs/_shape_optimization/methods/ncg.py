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

"""Nonlinear conjugate gradient methods for shape optimization.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import numpy as np

from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm


if TYPE_CHECKING:
    from ..shape_optimization_problem import ShapeOptimizationProblem


class NCG(ShapeOptimizationAlgorithm):
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

        self.gradient_prev = fenics.Function(self.form_handler.deformation_space)
        self.difference = fenics.Function(self.form_handler.deformation_space)
        self.temp_HZ = fenics.Function(self.form_handler.deformation_space)

        self.cg_method = self.config.get("AlgoCG", "cg_method", fallback="FR")
        self.cg_periodic_restart = self.config.getboolean(
            "AlgoCG", "cg_periodic_restart", fallback=False
        )
        self.cg_periodic_its = self.config.getint(
            "AlgoCG", "cg_periodic_its", fallback=10
        )
        self.cg_relative_restart = self.config.getboolean(
            "AlgoCG", "cg_relative_restart", fallback=False
        )
        self.cg_restart_tol = self.config.getfloat(
            "AlgoCG", "cg_restart_tol", fallback=0.25
        )

    def run(self) -> None:
        """Performs the optimization via the nonlinear cg method

        Returns
        -------
        None
        """

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
        self.memory = 0
        self.relative_norm = 1.0
        self.state_problem.has_solution = False
        self.gradient.vector().vec().set(1.0)

        while True:

            self.gradient_prev.vector().vec().aypx(0.0, self.gradient.vector().vec())

            self.adjoint_problem.has_solution = False
            self.shape_gradient_problem.has_solution = False
            self.shape_gradient_problem.solve()

            self.gradient_norm = np.sqrt(
                self.shape_gradient_problem.gradient_norm_squared
            )
            if self.iteration > 0:
                if self.cg_method == "FR":
                    self.beta_numerator = self.form_handler.scalar_product(
                        self.gradient, self.gradient
                    )
                    self.beta_denominator = self.form_handler.scalar_product(
                        self.gradient_prev, self.gradient_prev
                    )
                    self.beta = self.beta_numerator / self.beta_denominator

                elif self.cg_method == "PR":
                    self.difference.vector().vec().aypx(
                        0.0,
                        self.gradient.vector().vec()
                        - self.gradient_prev.vector().vec(),
                    )

                    self.beta_numerator = self.form_handler.scalar_product(
                        self.gradient, self.difference
                    )
                    self.beta_denominator = self.form_handler.scalar_product(
                        self.gradient_prev, self.gradient_prev
                    )
                    self.beta = self.beta_numerator / self.beta_denominator
                    # self.beta = np.maximum(self.beta, 0.0)

                elif self.cg_method == "HS":
                    self.difference.vector().vec().aypx(
                        0.0,
                        self.gradient.vector().vec()
                        - self.gradient_prev.vector().vec(),
                    )

                    self.beta_numerator = self.form_handler.scalar_product(
                        self.gradient, self.difference
                    )
                    self.beta_denominator = self.form_handler.scalar_product(
                        self.difference, self.search_direction
                    )
                    self.beta = self.beta_numerator / self.beta_denominator

                elif self.cg_method == "DY":
                    self.difference.vector().vec().aypx(
                        0.0,
                        self.gradient.vector().vec()
                        - self.gradient_prev.vector().vec(),
                    )

                    self.beta_numerator = self.form_handler.scalar_product(
                        self.gradient, self.gradient
                    )
                    self.beta_denominator = self.form_handler.scalar_product(
                        self.difference, self.search_direction
                    )
                    self.beta = self.beta_numerator / self.beta_denominator

                elif self.cg_method == "HZ":
                    self.difference.vector().vec().aypx(
                        0.0,
                        self.gradient.vector().vec()
                        - self.gradient_prev.vector().vec(),
                    )

                    dy = self.form_handler.scalar_product(
                        self.difference, self.search_direction
                    )
                    y2 = self.form_handler.scalar_product(
                        self.difference, self.difference
                    )

                    self.difference.vector().vec().axpy(
                        -2 * y2 / dy, self.search_direction.vector().vec()
                    )

                    self.beta = (
                        self.form_handler.scalar_product(self.gradient, self.difference)
                        / dy
                    )

            if self.iteration == 0:
                self.gradient_norm_initial = self.gradient_norm
                if self.gradient_norm_initial == 0:
                    self.converged = True
                    break
                self.beta = 0.0

            self.relative_norm = self.gradient_norm / self.gradient_norm_initial
            if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
                self.converged = True
                break

            self.search_direction.vector().vec().aypx(
                self.beta, -self.gradient.vector().vec()
            )
            if self.cg_periodic_restart:
                if self.memory < self.cg_periodic_its:
                    self.memory += 1
                else:
                    self.search_direction.vector().vec().aypx(
                        0.0, -self.gradient.vector().vec()
                    )
                    self.memory = 0
            if self.cg_relative_restart:
                if (
                    abs(
                        self.form_handler.scalar_product(
                            self.gradient, self.gradient_prev
                        )
                    )
                    / pow(self.gradient_norm, 2)
                    >= self.cg_restart_tol
                ):
                    self.search_direction.vector().vec().aypx(
                        0.0, -self.gradient.vector().vec()
                    )
                    self.memory = 0

            self.directional_derivative = self.form_handler.scalar_product(
                self.gradient, self.search_direction
            )

            if self.directional_derivative >= 0:
                self.search_direction.vector().vec().aypx(
                    0.0, -self.gradient.vector().vec()
                )

            self.line_search.search(self.search_direction, self.has_curvature_info)

            self.iteration += 1
            if self.nonconvergence():
                break
