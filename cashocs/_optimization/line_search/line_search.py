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

"""Module for the parent class of all line searches."""

from __future__ import annotations

import abc
from typing import List, TYPE_CHECKING

import fenics

from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import types
    from cashocs._optimization import optimization_algorithms


class LineSearch(abc.ABC):
    """Abstract implementation of a line search."""

    def __init__(self, optimization_problem: types.OptimizationProblem) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.

        """
        self.config = optimization_problem.config
        self.form_handler = optimization_problem.form_handler
        self.gradient = optimization_problem.gradient
        self.state_problem = optimization_problem.state_problem
        self.optimization_variable_abstractions = (
            optimization_problem.optimization_variable_abstractions
        )
        self.cost_functional = optimization_problem.reduced_cost_functional

        self.is_shape_problem = optimization_problem.is_shape_problem
        self.is_control_problem = optimization_problem.is_control_problem

        self.stepsize = self.config.getfloat("OptimizationRoutine", "initial_stepsize")
        self.safeguard_stepsize = self.config.getboolean(
            "OptimizationRoutine", "safeguard_stepsize"
        )

        algorithm = _utils.optimization_algorithm_configuration(self.config)
        self.is_newton_like = algorithm.casefold() == "lbfgs"
        self.is_newton = algorithm.casefold() == "newton"
        self.is_steepest_descent = algorithm.casefold() == "gradient_descent"
        if self.is_newton:
            self.stepsize = 1.0

    def perform(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
    ) -> None:
        """Performs a line search for the new iterate.

        Notes:
            This is the function that should be called in the optimization algorithm,
            it consists of a call to ``self.search`` and ``self.post_line_search``
            afterwards.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates, whether the search direction
                is (presumably) scaled.

        """
        self.search(solver, search_direction, has_curvature_info)
        self.post_line_search()

    @abc.abstractmethod
    def search(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
    ) -> None:
        """Performs a line search.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates, whether the search direction
                is (presumably) scaled.

        """
        pass

    def post_line_search(self) -> None:
        """Performs tasks after the line search was successful."""
        self.form_handler.update_scalar_product()
