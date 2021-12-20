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

"""Line search for optimal control problems.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import fenics
import numpy as np

from .._interfaces.line_search import LineSearch
from .._loggers import error


if TYPE_CHECKING:
    from .control_optimization_algorithm import ControlOptimizationAlgorithm


class ArmijoLineSearch(LineSearch):
    """An Armijo-based line search for optimal control

    Implements an Armijo line search for the solution of control problems.
    The exact behavior can be controlled via the config file.
    """

    def __init__(self, optimization_algorithm: ControlOptimizationAlgorithm) -> None:
        """
        Parameters
        ----------
        optimization_algorithm : ControlOptimizationAlgorithm
            The corresponding optimization algorihm
        """

        super().__init__(optimization_algorithm)

        self.projected_difference = [
            fenics.Function(V) for V in self.form_handler.control_spaces
        ]
        self.controls = optimization_algorithm.controls
        self.controls_temp = optimization_algorithm.controls_temp
        self.gradients = optimization_algorithm.gradients

    def decrease_measure(
        self, search_direction: Optional[fenics.Function] = None
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test

        Parameters
        ----------
        search_direction : fenics.Function or None, optional
            The search direction (not required)

        Returns
        -------
        float
            The decrease measure for the Armijo test
        """

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0,
                self.controls[j].vector().vec() - self.controls_temp[j].vector().vec(),
            )

        return self.form_handler.scalar_product(
            self.gradients, self.projected_difference
        )

    def search(
        self, search_directions: List[fenics.Function], has_curvature_info: bool
    ) -> None:
        """Does a line search with the Armijo rule.

        Performs the line search along the entered search direction and adapts
        the step size if curvature information is contained in the search direction.

        Parameters
        ----------
        search_directions : list[fenics.Function]
            The current search direction computed by the optimization algorithm
        has_curvature_info : bool
            A boolean flag, indicating whether the search direction is (actually)
            computed by a BFGS or Newton method

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

        if has_curvature_info:
            self.stepsize = 1.0

        self.ref_algo().print_results()

        for j in range(self.form_handler.control_dim):
            self.controls_temp[j].vector().vec().aypx(
                0.0, self.controls[j].vector().vec()
            )

        while True:
            if self.stepsize * self.search_direction_inf <= 1e-8:
                error("Stepsize too small.")
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
                error("Stepsize too small.")
                for j in range(self.form_handler.control_dim):
                    self.controls[j].vector().vec().aypx(
                        0.0, self.controls_temp[j].vector().vec()
                    )
                break

            for j in range(len(self.controls)):
                self.controls[j].vector().vec().axpy(
                    self.stepsize, search_directions[j].vector().vec()
                )

            self.form_handler.project_to_admissible_set(self.controls)

            self.ref_algo().state_problem.has_solution = False
            self.objective_step = self.cost_functional.evaluate()

            # self.project_direction_active(search_directions)
            # meas = -self.epsilon_armijo*self.stepsize*self.form_handler.scalar_product(self.gradients, self.directions)

            if (
                self.objective_step
                < self.ref_algo().objective_value
                + self.epsilon_armijo * self.decrease_measure()
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

        if not self.ref_algo().line_search_broken:
            self.ref_algo().stepsize = self.stepsize
            self.ref_algo().objective_value = self.objective_step

        if not has_curvature_info:
            self.stepsize *= self.beta_armijo
