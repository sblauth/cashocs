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

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import fenics
import numpy as np

from ..optimization_variable_handler import OptimizationVariableHandler


if TYPE_CHECKING:
    from shape_optimization_problem import (
        ShapeOptimizationProblem,
    )


class ShapeVariableHandler(OptimizationVariableHandler):
    def __init__(self, optimization_problem: ShapeOptimizationProblem) -> None:

        super().__init__(optimization_problem)
        self.mesh_handler = optimization_problem.mesh_handler
        self.deformation = fenics.Function(self.form_handler.deformation_space)

        temp_dict = optimization_problem.temp_dict
        if self.mesh_handler.do_remesh:
            optimization_problem.output_manager.set_remesh(
                temp_dict.get("remesh_counter", 0)
            )

    def compute_decrease_measure(
        self, search_direction: Optional[List[fenics.Function]] = None
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test

        Parameters
        ----------
        search_direction : list[fenics.Function]
            The current search direction

        Returns
        -------
        float
            the decrease measure for the Armijo rule
        """

        return self.form_handler.scalar_product(self.gradient, search_direction)

    def compute_gradient_norm(self) -> float:

        return np.sqrt(self.form_handler.scalar_product(self.gradient, self.gradient))

    def revert_variable_update(self) -> None:

        self.mesh_handler.revert_transformation()

    def update_optimization_variables(
        self, search_direction, stepsize: float, beta: float
    ) -> float:

        while True:
            self.deformation.vector().vec().aypx(
                0.0, stepsize * search_direction[0].vector().vec()
            )
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
    ) -> float:

        return self.mesh_handler.compute_decreases(search_direction, stepsize)

    def requires_remeshing(self) -> bool:

        if (
            self.mesh_handler.current_mesh_quality
            < self.mesh_handler.mesh_quality_tol_upper
        ):
            return True
        else:
            return False

    def project_ncg_search_direction(
        self, search_direction: List[fenics.Function]
    ) -> None:

        pass
