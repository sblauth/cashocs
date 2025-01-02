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

"""Database for functions."""

from __future__ import annotations

import fenics

from cashocs import _utils


class FunctionDatabase:
    """The database for functions and function spaces."""

    def __init__(
        self,
        states: list[fenics.Function],
        adjoints: list[fenics.Function],
    ) -> None:
        """Initializes the database.

        Args:
            states: The list of state variables.
            adjoints: The list of adjoint variables.

        """
        self.states = states
        self.adjoints = adjoints

        self.state_spaces = [x.function_space() for x in self.states]
        self.adjoint_spaces = [x.function_space() for x in self.adjoints]
        self.trial_functions_state = [
            fenics.TrialFunction(function_space) for function_space in self.state_spaces
        ]
        self.test_functions_state = [
            fenics.TestFunction(function_space) for function_space in self.state_spaces
        ]
        self.trial_functions_adjoint = [
            fenics.TrialFunction(function_space)
            for function_space in self.adjoint_spaces
        ]
        self.test_functions_adjoint = [
            fenics.TestFunction(function_space)
            for function_space in self.adjoint_spaces
        ]

        mesh = self.state_spaces[0].mesh()
        self.cg_function_space = fenics.FunctionSpace(mesh, "CG", 1)
        self.dg_function_space = fenics.FunctionSpace(mesh, "DG", 0)

        self.states_prime = _utils.create_function_list(self.state_spaces)
        self.adjoints_prime = _utils.create_function_list(self.adjoint_spaces)

        self.control_spaces: list[fenics.FunctionSpace] = []
        self.gradient: list[fenics.Function] = []

        self.controls: list[fenics.Function] = []
