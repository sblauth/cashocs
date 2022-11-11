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

"""Main Database for all of cashocs."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    import fenics
    import ufl

    from cashocs import _typing


class FormDatabase:
    """Database for user supplied forms."""

    def __init__(
        self,
        cost_functional_list: List[_typing.CostFunctional],
        state_forms: List[ufl.Form],
        bcs_list: List[List[fenics.DirichletBC]],
    ):
        """Initializes the form database.

        Args:
            cost_functional_list: The list of cost functionals.
            state_forms: The list of state forms.
            bcs_list: The list of boundary conditions for the state system.

        """
        self.cost_functional_list = cost_functional_list
        self.state_forms = state_forms
        self.bcs_list = bcs_list

        self.lagrangian = cost_functional.Lagrangian(
            self.cost_functional_list, self.state_forms
        )
