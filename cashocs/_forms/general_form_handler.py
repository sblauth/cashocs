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

"""Management for weak forms for general optimization problems."""

from __future__ import annotations

import fenics

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _utils
from cashocs._database import database


def _get_subdx(
    function_space: fenics.FunctionSpace, index: int, ls: list
) -> None | list[int]:
    """Computes the sub-indices for mixed function spaces based on the id of a subspace.

    Args:
        function_space: The function space, whose substructure is to be investigated.
        index: The id of the target function space.
        ls: A list of indices for the sub-spaces.

    Returns:
        The list of the sub-indices.

    """
    if function_space.id() == index:
        return ls
    if function_space.num_sub_spaces() > 1:
        for i in range(function_space.num_sub_spaces()):
            ans = _get_subdx(function_space.sub(i), index, ls + [i])
            if ans is not None:
                return ans

    return None


class GeneralFormHandler:
    """Manages weak state and adjoint forms."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        self.db = db

        self.config = self.db.config
        self.state_form_handler: StateFormHandler = StateFormHandler(self.db)
        self.adjoint_form_handler: AdjointFormHandler = AdjointFormHandler(self.db)


class StateFormHandler:
    """Manages weak state forms."""

    state_eq_forms: list[ufl.Form]
    linear_state_eq_forms: list[ufl.Form]
    state_eq_forms_lhs: list[ufl.Form]
    state_eq_forms_rhs: list[ufl.Form]

    def __init__(self, db: database.Database) -> None:
        """Initializes the state form handler.

        Args:
            db: The database of the problem.

        """
        self.db = db

        self.config = self.db.config
        self.bcs_list: list[list[fenics.DirichletBC]] = self.db.form_db.bcs_list
        (
            self.state_eq_forms,
            self.linear_state_eq_forms,
            self.state_eq_forms_lhs,
            self.state_eq_forms_rhs,
        ) = self._compute_state_equations()

    def _compute_state_equations(
        self,
    ) -> tuple[list[ufl.Form], list[ufl.Form], list[ufl.Form], list[ufl.Form]]:
        """Calculates the weak form of the state equation for the use with fenics."""
        state_eq_forms = [
            fenics.derivative(
                self.db.form_db.state_forms[i],
                self.db.function_db.adjoints[i],
                self.db.function_db.test_functions_state[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]
        linear_state_eq_forms = []
        state_eq_forms_lhs: list[ufl.Form] = []
        state_eq_forms_rhs: list[ufl.Form] = []

        if self.config.getboolean("StateSystem", "is_linear"):
            linear_state_eq_forms = [
                ufl.replace(
                    state_eq_forms[i],
                    {
                        self.db.function_db.states[
                            i
                        ]: self.db.function_db.trial_functions_state[i]
                    },
                )
                for i in range(self.db.parameter_db.state_dim)
            ]

            (
                state_eq_forms_lhs,
                state_eq_forms_rhs,
            ) = _utils.split_linear_forms(linear_state_eq_forms)

        return (
            state_eq_forms,
            linear_state_eq_forms,
            state_eq_forms_lhs,
            state_eq_forms_rhs,
        )


class AdjointFormHandler:
    """Manages weak adjoint forms."""

    bcs_list_ad: list[list[fenics.DirichletBC]]
    adjoint_eq_lhs: list[ufl.Form]
    adjoint_eq_rhs: list[ufl.Form]
    adjoint_eq_forms: list[ufl.Form]
    linear_adjoint_eq_forms: list[ufl.Form]

    def __init__(self, db: database.Database) -> None:
        """Initializes the adjoint form handler.

        Args:
            db: The database of the problem.

        """
        self.db = db

        self.config = self.db.config
        self.bcs_list_ad = self._compute_adjoint_boundary_conditions()
        (
            self.adjoint_eq_forms,
            self.linear_adjoint_eq_forms,
            self.adjoint_eq_lhs,
            self.adjoint_eq_rhs,
        ) = self._compute_adjoint_equations()

    def _compute_adjoint_boundary_conditions(self) -> list[list[fenics.DirichletBC]]:
        """Computes the boundary conditions for the adjoint systems."""
        if self.db.parameter_db.state_adjoint_equal_spaces:
            bcs_list_ad = [
                [fenics.DirichletBC(bc) for bc in self.db.form_db.bcs_list[i]]
                for i in range(self.db.parameter_db.state_dim)
            ]
            for i in range(self.db.parameter_db.state_dim):
                for bc in bcs_list_ad[i]:
                    bc.homogenize()
        else:
            bcs_list_ad = [
                [1] * len(self.db.form_db.bcs_list[i])
                for i in range(self.db.parameter_db.state_dim)
            ]

            for i in range(self.db.parameter_db.state_dim):
                for j, bc in enumerate(self.db.form_db.bcs_list[i]):
                    idx = bc.function_space().id()
                    subdx: list[int] = []
                    _get_subdx(self.db.function_db.state_spaces[i], idx, ls=subdx)
                    adjoint_space = self.db.function_db.adjoint_spaces[i]
                    for num in subdx:
                        adjoint_space = adjoint_space.sub(num)
                    shape = adjoint_space.ufl_element().value_shape()
                    if shape == ():
                        bdry_value = fenics.Constant(0)
                    else:
                        bdry_value = fenics.Constant(
                            [0] * adjoint_space.ufl_element().value_size()
                        )

                    try:
                        bcs_list_ad[i][j] = fenics.DirichletBC(
                            adjoint_space,
                            bdry_value,
                            bc.domain_args[0],
                            bc.domain_args[1],
                        )
                    except AttributeError:
                        bcs_list_ad[i][j] = fenics.DirichletBC(
                            adjoint_space, bdry_value, bc.sub_domain
                        )

        return bcs_list_ad

    def _compute_adjoint_equations(
        self,
    ) -> tuple[list[ufl.Form], list[ufl.Form], list[ufl.Form], list[ufl.Form]]:
        """Calculates the weak form of the adjoint equation for use with fenics."""
        adjoint_eq_forms = [
            self.db.form_db.lagrangian.derivative(
                self.db.function_db.states[i],
                self.db.function_db.test_functions_adjoint[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        linear_adjoint_eq_forms = [
            ufl.replace(
                adjoint_eq_forms[i],
                {
                    self.db.function_db.adjoints[
                        i
                    ]: self.db.function_db.trial_functions_adjoint[i]
                },
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        adjoint_eq_lhs, adjoint_eq_rhs = _utils.split_linear_forms(
            linear_adjoint_eq_forms
        )

        return adjoint_eq_forms, linear_adjoint_eq_forms, adjoint_eq_lhs, adjoint_eq_rhs
