# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Implementation of a reduced cost functional.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .._interfaces import ReducedCostFunctional


if TYPE_CHECKING:
    from .._forms import ControlFormHandler
    from .._pde_problems import StateProblem


class ReducedControlCostFunctional(ReducedCostFunctional):
    """The reduced cost functional for the optimization problem

    A class that represents an reduced cost functional of an optimal control problem, which
    is used to evaluate it.
    """

    def __init__(
        self, form_handler: ControlFormHandler, state_problem: StateProblem
    ) -> None:
        """Initialize the reduced cost functional

        Parameters
        ----------
        form_handler : ControlFormHandler
            The FormHandler object for the optimization problem
        state_problem : StateProblem
            The StateProblem object corresponding to the state system
        """

        super().__init__(form_handler, state_problem)

    def evaluate(self) -> float:
        """Evaluates the reduced cost functional.

        First solves the state system, so that the state variables are up-to-date,
        and then evaluates the reduced cost functional by assembling the corresponding
        UFL form.

        Returns
        -------
        float
            The value of the reduced cost functional
        """

        return super().evaluate()
