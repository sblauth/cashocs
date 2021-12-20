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

"""Line search for inner PDAS solvers.

"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, List

import fenics
import numpy as np


if TYPE_CHECKING:
    from ...control_optimization_algorithm import ControlOptimizationAlgorithm


class UnconstrainedLineSearch:
    """Armijo line search for unconstrained optimization problems"""

    def __init__(self, optimization_algorithm: ControlOptimizationAlgorithm) -> None:
        """Initializes the line search

        Parameters
        ----------
        optimization_algorithm : ControlOptimizationAlgorithm
            the corresponding optimization algorithm
        """

        self.ref_algo = weakref.ref(optimization_algorithm)
        self.config = optimization_algorithm.config
        self.form_handler = optimization_algorithm.form_handler

        self.stepsize = self.config.getfloat(
            "OptimizationRoutine", "initial_stepsize", fallback=1.0
        )
        self.epsilon_armijo = self.config.getfloat(
            "OptimizationRoutine", "epsilon_armijo", fallback=1e-4
        )
        self.beta_armijo = self.config.getfloat(
            "OptimizationRoutine", "beta_armijo", fallback=2.0
        )
        self.armijo_stepsize_initial = self.stepsize

        self.controls_temp = optimization_algorithm.controls_temp
        self.cost_functional = optimization_algorithm.cost_functional

        self.controls = optimization_algorithm.controls
        self.gradients = optimization_algorithm.gradients

        inner_pdas = self.config.get("AlgoPDAS", "inner_pdas")
        self.is_newton_like = inner_pdas in ["lbfgs", "bfgs"]
        self.is_newton = inner_pdas in ["newton"]
        self.is_steepest_descent = inner_pdas in ["gradient_descent", "gd"]
        if self.is_newton:
            self.stepsize = 1.0

    def decrease_measure(self, search_directions: List[fenics.Function]) -> float:
        """Computes the decrease measure for the Armijo rule

        Parameters
        ----------
        search_directions : list[fenics.Function]
            The search direction computed by optimization_algorithm

        Returns
        -------
        float
            The decrease measure for the Armijo rule
        """

        return self.stepsize * self.form_handler.scalar_product(
            self.gradients, search_directions
        )

    def search(self, search_directions: List[fenics.Function]) -> None:
        """Performs an Armijo line search

        Parameters
        ----------
        search_directions : list[fenics.Function]
            The search direction computed by the optimization_algorithm

        Returns
        -------
        None
        """

        self.search_direction_inf = np.max(
            [
                np.max(np.abs(search_directions[i].vector()[:]))
                for i in range(len(self.gradients))
            ]
        )
        self.ref_algo().objective_value = self.cost_functional.evaluate()

        # self.ref_algo().print_results()

        for j in range(self.form_handler.control_dim):
            self.controls_temp[j].vector().vec().aypx(
                0.0, self.controls[j].vector().vec()
            )

        while True:
            if self.stepsize * self.search_direction_inf <= 1e-8:
                self.ref_algo().line_search_broken = True
                for j in range(self.form_handler.control_dim):
                    self.controls[j].vector().vec().aypx(
                        0.0, self.controls_temp[j].vector().vec()
                    )
                break
            elif (
                not self.is_newton_like
                and not self.is_newton
                and self.stepsize / self.armijo_stepsize_initial <= 1e-8
            ):
                self.ref_algo().line_search_broken = True
                for j in range(self.form_handler.control_dim):
                    self.controls[j].vector().vec().aypx(
                        0.0, self.controls_temp[j].vector().vec()
                    )
                break

            for j in range(len(self.controls)):
                self.controls[j].vector().vec().axpy(
                    self.stepsize, search_directions[j].vector().vec()
                )

            self.ref_algo().state_problem.has_solution = False
            self.objective_step = self.cost_functional.evaluate()

            if (
                self.objective_step
                < self.ref_algo().objective_value
                + self.epsilon_armijo * self.decrease_measure(search_directions)
            ):
                if self.ref_algo().iteration == 0:
                    self.armijo_stepsize_initial = self.stepsize
                break

            else:
                self.stepsize /= self.beta_armijo
                for i in range(len(self.controls)):
                    self.controls[i].vector().vec().aypx(
                        0.0, self.controls_temp[i].vector().vec()
                    )

        self.ref_algo().stepsize = self.stepsize
        self.ref_algo().objective_value = self.objective_step
        if not self.is_newton_like and not self.is_newton:
            self.stepsize *= self.beta_armijo
        else:
            self.stepsize = 1.0
