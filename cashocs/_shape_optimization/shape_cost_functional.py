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

"""Implementation of a reduced shape cost functional

"""

from .._interfaces import ReducedCostFunctional


class ReducedShapeCostFunctional(ReducedCostFunctional):
    """Reduced cost functional for a shape optimization problem"""

    def __init__(self, form_handler, state_problem):
        """Initializes the reduced cost functional

        Parameters
        ----------
        form_handler : cashocs._forms.ShapeFormHandler
                the ControlFormHandler object for the optimization problem
        state_problem : cashocs._pde_problems.StateProblem
                the StateProblem object corresponding to the state system
        """

        super().__init__(form_handler, state_problem)

    def evaluate(self):
        """Evaluates the reduced cost functional.

        First solves the state system, so that the state variables are up-to-date,
        and then evaluates the reduced cost functional by assembling the corresponding
        UFL form.

        Returns
        -------
        float
                the value of the reduced cost functional
        """

        val = super().evaluate()
        val_reg = self.form_handler.regularization.compute_objective()

        return val + val_reg
