# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

"""Management of optimization variables.

This is used to update, restore, and manipulate optimization with abstractions, so that
the same optimization algorithms can be used for different _typing of problems.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics
import numpy as np

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import geometry
    from cashocs._database import database


class OptimizationVariableAbstractions(abc.ABC):
    """Base class for abstracting optimization variables."""

    mesh_handler: geometry._MeshHandler  # pylint: disable=protected-access

    def __init__(
        self, optimization_problem: _typing.OptimizationProblem, db: database.Database
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            db: The database of the problem.

        """
        self.db = db

        self.form_handler = optimization_problem.form_handler
        self.deformation = fenics.Function(self.db.function_db.control_spaces[0])

    @abc.abstractmethod
    def compute_decrease_measure(
        self, search_direction: list[fenics.Function]
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
        self,
        search_direction: list[fenics.Function],
        stepsize: float,
        beta: float,
        active_idx: np.ndarray | None = None,
        constraint_gradient: np.ndarray | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> float:
        """Updates the optimization variables based on a line search.

        Args:
            search_direction: The current search direction.
            stepsize: The current (trial) stepsize.
            beta: The parameter for the line search, which "halves" the stepsize if the
                test was not successful.
            active_idx: The list of active indices of the working set. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.
            constraint_gradient: The gradient of the constraints for the mesh quality.
                Only needed for shape optimization with mesh quality constraints.
                Default is `None`.
            dropped_idx: The list of indicies for dropped constraints. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.

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
        self, search_direction: list[fenics.Function], stepsize: float
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
        self, search_direction: list[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """
        pass

    @abc.abstractmethod
    def compute_active_sets(self) -> None:
        """Computes the active sets of the problem."""
        pass

    def restrict_to_inactive_set(
        self, a: list[fenics.Function], b: list[fenics.Function]
    ) -> list[fenics.Function]:
        """Restricts a function to the inactive set.

        Note, that nothing will happen if the type of the optimization problem does not
        support box constraints.

        Args:
            a: The function, which shall be restricted to the inactive set
            b: The output function, which will contain the result and is overridden.

        Returns:
            The result of the restriction (overrides input b)

        """
        for j in range(len(b)):
            if not b[j].vector().vec().equal(a[j].vector().vec()):
                b[j].vector().vec().aypx(0.0, a[j].vector().vec())
                b[j].vector().apply("")

        return b

    # pylint: disable=unused-argument
    def restrict_to_active_set(
        self, a: list[fenics.Function], b: list[fenics.Function]
    ) -> list[fenics.Function]:
        """Restricts a function to the active set.

        Note, that nothing will happen if the type of the optimization problem does not
        support box constraints.

        Args:
            a: The function, which shall be restricted to the active set
            b: The output function, which will contain the result and is overridden.

        Returns:
            The result of the restriction (overrides input b)

        """
        for j in range(len(b)):
            b[j].vector().vec().set(0.0)
            b[j].vector().apply("")

        return b
