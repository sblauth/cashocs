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

"""Database for geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics

from cashocs import _utils

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

if TYPE_CHECKING:
    from mpi4py import MPI
    from petsc4py import PETSc

    from cashocs._database import function_database
    from cashocs._database import parameter_database


class GeometryDatabase:
    """Database for geometry parameters."""

    transfer_matrix: PETSc.Mat
    old_transfer_matrix: PETSc.Mat

    def __init__(
        self,
        function_db: function_database.FunctionDatabase,
        parameter_db: parameter_database.ParameterDatabase,
    ) -> None:
        """Initializes the geometry database.

        Args:
            function_db: The database for function parameters.
            parameter_db: The database for other parameters.

        """
        self.mesh: fenics.Mesh = function_db.state_spaces[0].mesh()
        self.dx: ufl.Measure = ufl.Measure("dx", self.mesh)
        self.mpi_comm: MPI.Comm = self.mesh.mpi_comm()

        self.function_db = function_db
        self.parameter_db = parameter_db

    def init_transfer_matrix(self) -> None:
        """Initializes the transfer matrix for computing the global deformation."""
        if self.parameter_db.temp_dict:
            self.transfer_matrix = self.parameter_db.temp_dict["transfer_matrix"].copy()
            self.old_transfer_matrix = self.parameter_db.temp_dict[
                "old_transfer_matrix"
            ].copy()
        else:
            interp = _utils.Interpolator(
                self.function_db.control_spaces[0], self.function_db.control_spaces[0]
            )
            self.transfer_matrix = interp.transfer_matrix
