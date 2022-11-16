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

"""General line search."""

from __future__ import annotations

import abc
from typing import List, TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import optimization_algorithms


class LineSearch(abc.ABC):
    """Abstract implementation of a line search."""

    def __init__(
        self, db: database.Database, optimization_problem: _typing.OptimizationProblem
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            optimization_problem: The corresponding optimization problem.

        """
        self.db = db

        self.config = self.db.config
        self.form_handler = optimization_problem.form_handler
        self.state_problem = optimization_problem.state_problem
        self.optimization_variable_abstractions = (
            optimization_problem.optimization_variable_abstractions
        )
        self.cost_functional = optimization_problem.reduced_cost_functional

        self.stepsize = self.config.getfloat("LineSearch", "initial_stepsize")
        self.safeguard_stepsize = self.config.getboolean(
            "LineSearch", "safeguard_stepsize"
        )

        self.beta_armijo: float = self.config.getfloat("LineSearch", "beta_armijo")
        self.search_direction_inf = 1.0

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

    def initialize_stepsize(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
    ) -> None:
        """Initializes the stepsize.

        Performs various ways for safeguarding (can be deactivated).

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates whether the direction is
                (presumably) scaled.

        """
        self.search_direction_inf = np.max(
            [
                search_direction[i].vector().norm("linf")
                for i in range(len(search_direction))
            ]
        )

        if has_curvature_info:
            self.stepsize = 1.0

        if solver.is_restarted:
            self.stepsize = self.config.getfloat("LineSearch", "initial_stepsize")

        num_decreases = (
            self.optimization_variable_abstractions.compute_a_priori_decreases(
                search_direction, self.stepsize
            )
        )
        self.stepsize /= pow(self.beta_armijo, num_decreases)

        if self.safeguard_stepsize and solver.iteration == 0:
            search_direction_norm = np.sqrt(
                self.form_handler.scalar_product(search_direction, search_direction)
            )
            self.stepsize = float(
                np.minimum(self.stepsize, 100.0 / (1.0 + search_direction_norm))
            )

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
