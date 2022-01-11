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

import abc
from typing import TYPE_CHECKING, List

import fenics

from ...utils import _optimization_algorithm_configuration


if TYPE_CHECKING:
    from ..optimization_problem import OptimizationProblem
    from ..optimization_algorithms import OptimizationAlgorithm


class LineSearch(abc.ABC):
    def __init__(self, optimization_problem: OptimizationProblem) -> None:

        self.config = optimization_problem.config
        self.form_handler = optimization_problem.form_handler
        self.gradient = optimization_problem.gradient
        self.state_problem = optimization_problem.state_problem
        self.optimization_variable_handler = (
            optimization_problem.optimization_variable_handler
        )
        self.cost_functional = optimization_problem.reduced_cost_functional

        self.is_shape_problem = optimization_problem.is_shape_problem
        self.is_control_problem = optimization_problem.is_control_problem

        self.stepsize = self.config.getfloat(
            "OptimizationRoutine", "initial_stepsize", fallback=1.0
        )

        algorithm = _optimization_algorithm_configuration(self.config)
        self.is_newton_like = algorithm == "lbfgs"
        self.is_newton = algorithm == "newton"
        self.is_steepest_descent = algorithm == "gradient_descent"
        if self.is_newton:
            self.stepsize = 1.0

    def perform(
        self,
        solver: OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
    ) -> None:

        self.search(solver, search_direction, has_curvature_info)
        self.post_line_search()

    @abc.abstractmethod
    def search(
        self, solver, search_direction: List[fenics.Function], has_curvature_info: bool
    ) -> None:
        pass

    def post_line_search(self) -> None:

        self.form_handler.update_scalar_product()