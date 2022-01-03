# Copyright (C) 2020-2022 Sebastian Blauth
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

"""Blueprint for the optimization algorithms.

"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics

from .._interfaces import OptimizationAlgorithm


if TYPE_CHECKING:
    from .optimal_control_problem import OptimalControlProblem


class ControlOptimizationAlgorithm(OptimizationAlgorithm):
    """Abstract class representing a optimization algorithm

    This is used for subclassing with the specific optimization methods
    later on.

    See Also
    --------
    methods.gradient_descent.GradientDescent
    methods.cg.CG
    methods.l_bfgs.LBFGS
    methods.newton.Newton
    methods.primal_dual_active_set_method.PDAS
    """

    def __init__(self, optimization_problem: OptimalControlProblem) -> None:
        """Initializes the optimization algorithm

        Defines common parameters used by all sub-classes.

        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The OptimalControlProblem class defined by the user
        """

        super().__init__(optimization_problem)

        self.controls = optimization_problem.controls
        self.controls_temp = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]
        self.projected_difference = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]
        self.search_direction = [
            fenics.Function(V) for V in optimization_problem.control_spaces
        ]

        self.require_control_constraints = (
            optimization_problem.require_control_constraints
        )

        self.pdas_solver = False

    @abc.abstractmethod
    def run(self) -> None:
        """Blueprint for a print function

        This is overwritten by the specific optimization algorithms later on.

        Returns
        -------
        None
        """

        pass

    def _stationary_measure_squared(self) -> float:
        """Computes the stationary measure (squared) corresponding to box-constraints

        In case there are no box constraints this reduces to the classical gradient
        norm.

        Returns
        -------
         float
            The square of the stationary measure

        """

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0, self.controls[j].vector().vec() - self.gradient[j].vector().vec()
            )

        self.form_handler.project_to_admissible_set(self.projected_difference)

        for j in range(self.form_handler.control_dim):
            self.projected_difference[j].vector().vec().aypx(
                0.0,
                self.controls[j].vector().vec()
                - self.projected_difference[j].vector().vec(),
            )

        return self.form_handler.scalar_product(
            self.projected_difference, self.projected_difference
        )
