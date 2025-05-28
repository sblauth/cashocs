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

"""General PDE problem."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics

from cashocs import _utils

if TYPE_CHECKING:
    from cashocs._database import database


class PDEProblem(abc.ABC):
    """Base class for a PDE problem."""

    def __init__(
        self,
        db: database.Database,
        linear_solver: _utils.linalg.LinearSolver | None = None,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem
            linear_solver: The linear solver (KSP) which is used to solve the linear
                systems arising from the discretized PDE

        """
        self.db = db

        self.config = db.config
        self.has_solution: bool = False
        if linear_solver is None:
            self.linear_solver = _utils.linalg.LinearSolver()
        else:
            self.linear_solver = linear_solver

    @abc.abstractmethod
    def solve(self) -> fenics.Function | list[fenics.Function]:
        """Solves the PDE.

        Returns:
            The solution of the PDE.

        """
        pass
