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

from typing import List, TYPE_CHECKING

import fenics

from .line_search import LineSearch
from .._loggers import error

if TYPE_CHECKING:
    from .._optimization_algorithms import OptimizationAlgorithm
    from .._interfaces import OptimizationProblem


import numpy as np


class ArmijoLineSearch(LineSearch):
    def __init__(
        self,
        optimization_problem: OptimizationProblem,
    ) -> None:

        super().__init__(optimization_problem)

        self.epsilon_armijo = self.config.getfloat(
            "OptimizationRoutine", "epsilon_armijo", fallback=1e-4
        )
        self.beta_armijo = self.config.getfloat(
            "OptimizationRoutine", "beta_armijo", fallback=2.0
        )
        self.armijo_stepsize_initial = self.stepsize

    def search(
        self,
        solver: OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
    ) -> int:

        self.search_direction_inf = np.max(
            [
                np.max(np.abs(search_direction[i].vector()[:]))
                for i in range(len(self.gradient))
            ]
        )

        if has_curvature_info:
            self.stepsize = 1.0

        num_decreases = self.optimization_variable_handler.compute_a_priori_decreases(
            search_direction, self.stepsize
        )
        self.stepsize /= pow(self.beta_armijo, num_decreases)

        while True:

            if solver.iteration >= solver.maximum_iterations:
                solver.remeshing_its = True
                exit_code = -4
                return exit_code

            if self.stepsize * self.search_direction_inf <= 1e-8:
                error("Stepsize too small.")
                # TODO: Investigate: Do we need this?
                self.optimization_variable_handler.revert_variable_update()
                solver.line_search_broken = True
                exit_code = -1
                return exit_code
            elif (
                not self.is_newton_like
                and not self.is_newton
                and self.stepsize / self.armijo_stepsize_initial <= 1e-8
            ):
                error("Stepsize too small.")
                # TODO: Investigate: Do we need this?
                self.optimization_variable_handler.revert_variable_update()
                solver.line_search_broken = True
                exit_code = -1
                return exit_code

            if self.is_shape_problem:
                decrease_measure_w_o_step = (
                    self.optimization_variable_handler.compute_decrease_measure(
                        search_direction
                    )
                )
            self.stepsize = (
                self.optimization_variable_handler.update_optimization_variables(
                    search_direction, self.stepsize, self.beta_armijo
                )
            )

            current_function_value = solver.objective_value

            self.state_problem.has_solution = False
            self.objective_step = self.cost_functional.evaluate()

            if self.is_control_problem:
                decrease_measure = (
                    self.optimization_variable_handler.compute_decrease_measure(
                        search_direction
                    )
                )
            if self.is_shape_problem:
                decrease_measure = decrease_measure_w_o_step * self.stepsize

            if (
                self.objective_step
                < current_function_value + self.epsilon_armijo * decrease_measure
            ):
                if self.optimization_variable_handler.requires_remeshing():
                    solver.requires_remeshing = True
                    exit_code = -3
                    return exit_code

                if solver.iteration == 0:
                    self.armijo_stepsize_initial = self.stepsize
                exit_code = 0
                break

            else:
                self.stepsize /= self.beta_armijo
                self.optimization_variable_handler.revert_variable_update()

        solver.stepsize = self.stepsize

        if not has_curvature_info:
            self.stepsize *= self.beta_armijo

        return exit_code
