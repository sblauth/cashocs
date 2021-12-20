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

"""Line search for shape optimization problems.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import numpy as np

from .._interfaces.line_search import LineSearch
from .._loggers import error


if TYPE_CHECKING:
    from .shape_optimization_algorithm import ShapeOptimizationAlgorithm


class ArmijoLineSearch(LineSearch):
    """An Armijo line search algorithm for shape optimization problems"""

    def __init__(self, optimization_algorithm: ShapeOptimizationAlgorithm) -> None:
        """Initializes the line search

        Parameters
        ----------
        optimization_algorithm : ShapeOptimizationAlgorithm
                the optimization problem of interest
        """

        super().__init__(optimization_algorithm)

        self.mesh_handler = optimization_algorithm.mesh_handler
        self.deformation = fenics.Function(self.form_handler.deformation_space)
        self.gradient = optimization_algorithm.gradient

    def decrease_measure(self, search_direction: fenics.Function) -> float:
        """Computes the measure of decrease needed for the Armijo test

        Parameters
        ----------
        search_direction : fenics.Function
            The current search direction

        Returns
        -------
        float
            the decrease measure for the Armijo rule
        """

        return self.form_handler.scalar_product(self.gradient, search_direction)

    def search(
        self, search_direction: fenics.Function, has_curvature_info: bool
    ) -> None:
        """Performs the line search along the entered search direction

        Parameters
        ----------
        search_direction : fenics.Function
            The current search direction computed by the algorithms
        has_curvature_info : bool
            ``True`` if the step is (actually) computed via L-BFGS or Newton

        Returns
        -------
        None
        """

        self.search_direction_inf = np.max(np.abs(search_direction.vector()[:]))
        self.ref_algo().objective_value = self.cost_functional.evaluate()

        if has_curvature_info:
            self.stepsize = 1.0

        self.ref_algo().print_results()

        num_decreases = self.mesh_handler.compute_decreases(
            search_direction, self.stepsize
        )
        self.stepsize /= pow(self.beta_armijo, num_decreases)

        while True:
            if self.ref_algo().iteration >= self.ref_algo().maximum_iterations:
                self.ref_algo().remeshing_its = True
                break

            if self.stepsize * self.search_direction_inf <= 1e-8:
                error("Stepsize too small.")
                self.ref_algo().line_search_broken = True
                break
            elif (
                not self.is_newton_like
                and self.stepsize / self.armijo_stepsize_initial <= 1e-8
            ):
                error("Stepsize too small.")
                self.ref_algo().line_search_broken = True
                break

            self.deformation.vector().vec().aypx(
                0.0, self.stepsize * search_direction.vector().vec()
            )
            self.dm = self.decrease_measure(search_direction)

            if self.mesh_handler.move_mesh(self.deformation):
                if (
                    self.mesh_handler.current_mesh_quality
                    < self.mesh_handler.mesh_quality_tol_lower
                ):
                    self.stepsize /= self.beta_armijo
                    self.mesh_handler.revert_transformation()
                    continue

                self.ref_algo().state_problem.has_solution = False
                self.objective_step = self.cost_functional.evaluate()

                if (
                    self.objective_step
                    < self.ref_algo().objective_value
                    + self.epsilon_armijo * self.stepsize * self.dm
                ):

                    if (
                        self.mesh_handler.current_mesh_quality
                        < self.mesh_handler.mesh_quality_tol_upper
                    ):
                        self.ref_algo().requires_remeshing = True
                        break

                    if self.ref_algo().iteration == 0:
                        self.armijo_stepsize_initial = self.stepsize
                    self.form_handler.update_scalar_product()
                    break

                else:
                    self.stepsize /= self.beta_armijo
                    self.mesh_handler.revert_transformation()

            else:
                self.stepsize /= self.beta_armijo

        if not (
            self.ref_algo().line_search_broken
            or self.ref_algo().requires_remeshing
            or self.ref_algo().remeshing_its
        ):
            self.ref_algo().stepsize = self.stepsize
            self.ref_algo().objective_value = self.objective_step

        if not has_curvature_info:
            self.stepsize *= self.beta_armijo
