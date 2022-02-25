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

"""Module for managing optimization variables.

This is used to update, restore, and manipulate optimization with abstractions, so that
the same optimization algorithms can be used for different types of problems.
"""

from __future__ import annotations

import abc
from typing import List, TYPE_CHECKING

import fenics

if TYPE_CHECKING:
    from cashocs import geometry
    from cashocs import types


class OptimizationVariableAbstractions(abc.ABC):
    """Base class for abstracting optimization variables."""

    mesh_handler: geometry._MeshHandler  # pylint: disable=protected-access

    def __init__(self, optimization_problem: types.OptimizationProblem) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.

        """
        self.gradient = optimization_problem.gradient
        self.form_handler = optimization_problem.form_handler

    @abc.abstractmethod
    def compute_decrease_measure(
        self, search_direction: List[fenics.Function]
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test.

        Args:
            search_direction: The search direction.

        Returns:
            The decrease measure for the Armijo test.

        """
        pass

    @abc.abstractmethod
    def revert_variable_update(self) -> None:
        """Reverts the optimization variables to the current iterate."""
        pass

    @abc.abstractmethod
    def update_optimization_variables(
        self, search_direction: List[fenics.Function], stepsize: float, beta: float
    ) -> float:
        """Updates the optimization variables based on a line search.

        Args:
            search_direction: The current search direction.
            stepsize: The current (trial) stepsize.
            beta: The parameter for the line search, which "halves" the stepsize if the
                test was not successful.

        Returns:
            The stepsize which was found to be acceptable.

        """
        pass

    @abc.abstractmethod
    def compute_gradient_norm(self) -> float:
        """Computes the norm of the gradient.

        Returns:
            The norm of the gradient.

        """
        pass

    @abc.abstractmethod
    def compute_a_priori_decreases(
        self, search_direction: List[fenics.Function], stepsize: float
    ) -> int:
        """Computes the number of times the stepsize has to be "halved" a priori.

        Args:
            search_direction: The current search direction.
            stepsize: The current stepsize.

        Returns:
            The number of times the stepsize has to be "halved" before the actual trial.

        """
        pass

    @abc.abstractmethod
    def requires_remeshing(self) -> bool:
        """Checks, if remeshing is needed.

        Returns:
            A boolean, which indicates whether remeshing is required.

        """
        pass

    @abc.abstractmethod
    def project_ncg_search_direction(
        self, search_direction: List[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """
        pass
