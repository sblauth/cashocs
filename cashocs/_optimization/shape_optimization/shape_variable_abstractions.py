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

"""Management of shape variables."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _forms
from cashocs import log
from cashocs._optimization import optimization_variable_abstractions

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import shape_optimization


class ShapeVariableAbstractions(
    optimization_variable_abstractions.OptimizationVariableAbstractions
):
    """Abstractions for optimization variables in the case of shape optimization."""

    def __init__(
        self,
        optimization_problem: shape_optimization.ShapeOptimizationProblem,
        db: database.Database,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            db: The database of the problem.

        """
        super().__init__(optimization_problem, db)
        self.form_handler = cast(_forms.ShapeFormHandler, self.form_handler)
        self.mesh_handler = optimization_problem.mesh_handler

    def compute_decrease_measure(
        self, search_direction: list[fenics.Function]
    ) -> float:
        """Computes the measure of decrease needed for the Armijo test.

        Args:
            search_direction: The search direction.

        Returns:
            The decrease measure for the Armijo test.

        """
        return self.form_handler.scalar_product(
            self.db.function_db.gradient, search_direction
        )

    def compute_gradient_norm(self) -> float:
        """Computes the norm of the gradient.

        Returns:
            The norm of the gradient.

        """
        res: float = np.sqrt(
            self.form_handler.scalar_product(
                self.db.function_db.gradient, self.db.function_db.gradient
            )
        )
        return res

    def revert_variable_update(self) -> None:
        """Reverts the optimization variables to the current iterate."""
        self.mesh_handler.revert_transformation()

    def update_optimization_variables(
        self,
        search_direction: list[fenics.Function],
        stepsize: float,
        beta: float,
        active_idx: np.ndarray | None = None,
        constraint_gradient: np.ndarray | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> float:
        """Updates the optimization variables based on a line search.

        Args:
            search_direction: The current search direction.
            stepsize: The current (trial) stepsize.
            beta: The parameter for the line search, which "halves" the stepsize if the
                test was not successful.
            active_idx: A boolean mask corresponding to the working set.
            constraint_gradient: The gradient of (all) constraints.
            dropped_idx: A boolean mask indicating which constraints have been recently
                dropped from the working set.

        Returns:
            The stepsize which was found to be acceptable.

        """
        while True:
            self.deformation.vector().vec().axpby(
                stepsize, 0.0, search_direction[0].vector().vec()
            )
            self.deformation.vector().apply("")
            log.debug(f"Stepsize for mesh update: {stepsize:.3e}")
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
        self, search_direction: list[fenics.Function], stepsize: float
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
        mesh_quality_criterion = bool(
            self.mesh_handler.current_mesh_quality
            < self.mesh_handler.mesh_quality_tol_upper
        )

        iteration = self.db.parameter_db.optimization_state["iteration"]
        if self.db.config.getint("MeshQuality", "remesh_iter") > 0:
            iteration_criterion = bool(
                iteration > 0
                and iteration % self.db.config.getint("MeshQuality", "remesh_iter") == 0
            )
        else:
            iteration_criterion = False

        requires_remeshing = mesh_quality_criterion or iteration_criterion
        return requires_remeshing

    def project_ncg_search_direction(
        self, search_direction: list[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """
        pass

    def compute_active_sets(self) -> None:
        """Computes the active sets of the problem."""
        pass
