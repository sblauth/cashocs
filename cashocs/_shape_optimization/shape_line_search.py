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

import weakref

import fenics
import numpy as np
from ufl import replace

from .._loggers import error
from ..utils import _optimization_algorithm_configuration


class ArmijoLineSearch:
    def __init__(self, optimization_algorithm):
        """Initializes the line search

        Parameters
        ----------
        optimization_algorithm : cashocs._shape_optimization.shape_optimization_algorithm.ShapeOptimizationAlgorithm
                the optimization problem of interest
        """

        self.ref_algo = weakref.ref(optimization_algorithm)
        self.config = optimization_algorithm.config
        self.form_handler = optimization_algorithm.form_handler
        self.mesh_handler = optimization_algorithm.mesh_handler
        self.deformation = fenics.Function(self.form_handler.deformation_space)

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

        self.cost_functional = optimization_algorithm.cost_functional

        self.gradient = optimization_algorithm.gradient

        self.algorithm = _optimization_algorithm_configuration(self.config)
        self.is_newton_like = self.algorithm == "lbfgs"
        self.is_steepest_descent = self.algorithm == "gradient_descent"

    def decrease_measure(self, search_direction):
        """Computes the measure of decrease needed for the Armijo test

        Parameters
        ----------
        search_direction : dolfin.function.function.Function
                The current search direction

        Returns
        -------
        float
                the decrease measure for the Armijo rule
        """

        return self.stepsize * self.form_handler.scalar_product(
            self.gradient, search_direction
        )
        # self.form = self.ref_algo().shape_gradient_problem.F_shape
        # self.form = replace(
        #     self.form, {self.form_handler.test_vector_field: -self.gradient}
        # )
        # return self.stepsize * fenics.assemble(self.form)

    def search(self, search_direction, has_curvature_info):
        """Performs the line search along the entered search direction

        Parameters
        ----------
        search_direction : dolfin.function.function.Function
                The current search direction computed by the algorithms
        has_curvature_info : bool
                True if the step is (actually) computed via L-BFGS or Newton

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

            self.deformation.vector()[:] = self.stepsize * search_direction.vector()[:]

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
                    + self.epsilon_armijo * self.decrease_measure(search_direction)
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
