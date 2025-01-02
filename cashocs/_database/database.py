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

"""Main Database for all of cashocs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics

from cashocs._database import form_database
from cashocs._database import function_database
from cashocs._database import geometry_database
from cashocs._database import parameter_database
from cashocs._optimization.optimization_algorithms import callback as cb

if TYPE_CHECKING:
    try:
        import ufl_legacy as ufl
    except ImportError:
        import ufl

    from cashocs import _typing
    from cashocs import io


class Database:
    """Database for many parameters."""

    def __init__(
        self,
        config: io.Config,
        states: list[fenics.Function],
        adjoints: list[fenics.Function],
        state_ksp_options: list[_typing.KspOption],
        adjoint_ksp_options: list[_typing.KspOption],
        gradient_ksp_options: list[_typing.KspOption] | None,
        cost_functional_list: list[_typing.CostFunctional],
        state_forms: list[ufl.Form],
        bcs_list: list[list[fenics.DirichletBC]],
        preconditioner_forms: list[ufl.Form],
    ) -> None:
        """Initialize the database.

        Args:
            config: The configuration for the problem.
            states: The list of state variables.
            adjoints: The list of adjoint variables.
            state_ksp_options: The list of ksp options for the state system.
            adjoint_ksp_options: The list of ksp options for the adjoint system.
            gradient_ksp_options: The list of ksp options for computing the gradient.
            cost_functional_list: The list of cost functionals.
            state_forms: The list of state forms.
            bcs_list: The list of Dirichlet boundary conditions for the state system.
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.

        """
        self.config = config
        self.callback = cb.Callback()

        self.function_db: function_database.FunctionDatabase = (
            function_database.FunctionDatabase(states, adjoints)
        )
        self.parameter_db: parameter_database.ParameterDatabase = (
            parameter_database.ParameterDatabase(
                self.function_db,
                config,
                state_ksp_options,
                adjoint_ksp_options,
                gradient_ksp_options,
            )
        )
        self.geometry_db: geometry_database.GeometryDatabase = (
            geometry_database.GeometryDatabase(self.function_db, self.parameter_db)
        )
        self.form_db: form_database.FormDatabase = form_database.FormDatabase(
            cost_functional_list, state_forms, bcs_list, preconditioner_forms
        )
