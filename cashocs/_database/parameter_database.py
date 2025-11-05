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

"""Database for parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cashocs import _exceptions
from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io
    from cashocs._database import function_database


class ParameterDatabase:
    """A database for many parameters."""

    def __init__(
        self,
        function_db: function_database.FunctionDatabase,
        config: io.Config,
        state_ksp_options: list[_typing.KspOption],
        adjoint_ksp_options: list[_typing.KspOption],
        gradient_ksp_options: list[_typing.KspOption] | None,
    ) -> None:
        """Initializes the database.

        Args:
            function_db: The database for functions.
            config: The configuration.
            state_ksp_options: The list of ksp options for the state system.
            adjoint_ksp_options: The list of ksp options for the adjoint system.
            gradient_ksp_options: The list of ksp options for computing the gradient.

        """
        self.config = config
        self.state_ksp_options = state_ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options

        self.gradient_ksp_options = gradient_ksp_options
        self.temp_dict: dict = {}

        self._problem_type = ""
        self.state_dim: int = len(function_db.states)

        self.display_box_constraints: bool = False

        self.state_adjoint_equal_spaces = False
        if function_db.state_spaces == function_db.adjoint_spaces:
            self.state_adjoint_equal_spaces = True

        self.opt_algo: str = _utils.optimization_algorithm_configuration(self.config)
        self.is_remeshed: bool = False

        self.control_dim: int = 1
        self.optimization_state: dict = {"stepsize": 1.0}
        self.remesh_directory: str = ""
        self.gmsh_file_path: str = ""

    @property
    def problem_type(self) -> str:
        """Returns the problem type."""
        return self._problem_type

    @problem_type.setter
    def problem_type(self, value: str) -> None:
        if value in ["control", "shape", "topology"]:
            self._problem_type = value
        else:
            raise _exceptions.InputError(
                "ParameterDatabase",
                "problem_type",
                "problem_type has to be either 'control' or 'shape'.",
            )
