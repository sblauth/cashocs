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

"""Abstractions of optimization variables in the case of shape optimization."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import fenics
import numpy as np

from cashocs._optimization import optimization_variable_abstractions

if TYPE_CHECKING:
    from cashocs._optimization import shape_optimization


class ShapeVariableAbstractions(
    optimization_variable_abstractions.OptimizationVariableAbstractions
):
    """Abstractions for optimization variables in the case of shape optimization."""

    def __init__(
        self, optimization_problem: shape_optimization.ShapeOptimizationProblem
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.

        """
        super().__init__(optimization_problem)
        self.mesh_handler = optimization_problem.mesh_handler
        self.deformation = fenics.Function(self.form_handler.deformation_space)

        if self.mesh_handler.do_remesh and optimization_problem.temp_dict is not None:
            temp_dict = optimization_problem.temp_dict
            optimization_problem.output_manager.set_remesh(
                temp_dict.get("remesh_counter", 0)
            )

    def compute_decrease_measure(
        self, search_direction: List[fenics.Function]
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test.

        Args:
            search_direction: The search direction.

        Returns:
            The decrease measure for the Armijo test.

        """
        return self.form_handler.scalar_product(self.gradient, search_direction)

    def compute_gradient_norm(self) -> float:
        """Computes the norm of the gradient.

        Returns:
            The norm of the gradient.

        """
        res: float = np.sqrt(
            self.form_handler.scalar_product(self.gradient, self.gradient)
        )
        return res

    def revert_variable_update(self) -> None:
        """Reverts the optimization variables to the current iterate."""
        self.mesh_handler.revert_transformation()

    def update_optimization_variables(
        self, search_direction: List[fenics.Function], stepsize: float, beta: float
    ) -> float:
        """Updates the optimization variables based on a line search.

        Args:
            search_direction: The current search direction.
            stepsize: The current (trial) stepsize.
            beta: The parameter for the line search, which "halves" the stepsize if the
                test was not successful.

        Returns:
            The stepsize which was found to be acceptable.

        """
        while True:
            self.deformation.vector().vec().aypx(
                0.0, stepsize * search_direction[0].vector().vec()
            )
            self.deformation.vector().apply("")
            if self.mesh_handler.move_mesh(self.deformation):
                if (
                    self.mesh_handler.current_mesh_quality
                    < self.mesh_handler.mesh_quality_tol_lower
                ):
                    stepsize /= beta
                    self.mesh_handler.revert_transformation()
                    continue
                else:
                    break
            else:
                stepsize /= beta

        return stepsize

    def compute_a_priori_decreases(
        self, search_direction: List[fenics.Function], stepsize: float
    ) -> int:
        """Computes the number of times the stepsize has to be "halved" a priori.

        Args:
            search_direction: The current search direction.
            stepsize: The current stepsize.

        Returns:
            The number of times the stepsize has to be "halved" before the actual trial.

        """
        return self.mesh_handler.compute_decreases(search_direction, stepsize)

    def requires_remeshing(self) -> bool:
        """Checks, if remeshing is needed.

        Returns:
            A boolean, which indicates whether remeshing is required.

        """
        return bool(
            self.mesh_handler.current_mesh_quality
            < self.mesh_handler.mesh_quality_tol_upper
        )

    def project_ncg_search_direction(
        self, search_direction: List[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """
        pass
