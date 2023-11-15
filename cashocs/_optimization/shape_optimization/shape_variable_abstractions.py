# Copyright (C) 2020-2023 Sebastian Blauth
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

from typing import cast, List, TYPE_CHECKING

import fenics
import numpy as np
from scipy import optimize

from cashocs import _forms
from cashocs._optimization import optimization_variable_abstractions

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import shape_optimization
    from cashocs._optimization import mesh_constraints


class ShapeVariableAbstractions(
    optimization_variable_abstractions.OptimizationVariableAbstractions
):
    """Abstractions for optimization variables in the case of shape optimization."""

    def __init__(
        self,
        optimization_problem: shape_optimization.ShapeOptimizationProblem,
        db: database.Database,
        constraint_manager: mesh_constraints.ConstraintManager,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            db: The database of the problem.

        """
        super().__init__(optimization_problem, db)
        self.form_handler = cast(_forms.ShapeFormHandler, self.form_handler)
        self.mesh_handler = optimization_problem.mesh_handler
        self.constraint_manager = constraint_manager
        self.mode = self.db.config.get("MeshQualityConstraints", "mode")

    def compute_decrease_measure(
        self, search_direction: List[fenics.Function]
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
            self.deformation.vector().vec().axpby(
                stepsize, 0.0, search_direction[0].vector().vec()
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

    def update_constrained_optimization_variables(
        self,
        search_direction: List[fenics.Function],
        stepsize: float,
        beta: float,
        active_idx,
        constraint_gradient,
        dropped_idx,
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
            coords_sequential = self.mesh_handler.mesh.coordinates().copy().reshape(-1)
            coords_dof = coords_sequential[self.constraint_manager.d2v]
            search_direction_dof = search_direction[0].vector()[:]

            if len(active_idx) > 0:
                coords_dof_feasible, stepsize = self.compute_step(
                    coords_dof,
                    search_direction_dof,
                    stepsize,
                    active_idx,
                    constraint_gradient,
                    dropped_idx,
                )

                dof_deformation_vector = coords_dof_feasible - coords_dof
                dof_deformation = fenics.Function(self.db.function_db.control_spaces[0])
                dof_deformation.vector()[:] = dof_deformation_vector

                self.deformation.vector().vec().axpby(
                    1.0, 0.0, dof_deformation.vector().vec()
                )
                self.deformation.vector().apply("")
            else:
                self.deformation.vector().vec().axpby(
                    stepsize, 0.0, search_direction[0].vector().vec()
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
                    # ToDo: Check for feasibility, re-project to working set
                    break
            else:
                stepsize /= beta

        return stepsize

    def project_to_working_set(
        self,
        coords_dof,
        search_direction_dof,
        stepsize,
        active_idx,
        constraint_gradient,
    ) -> np.ndarray:
        y_j = coords_dof + stepsize * search_direction_dof
        A = self.constraint_manager.compute_active_gradient(
            active_idx, constraint_gradient
        )
        if self.mode == "complete":
            S = self.form_handler.scalar_product_matrix[:, :]
            S_inv = np.linalg.inv(S)

        for i in range(10):
            if not np.all(
                self.constraint_manager.compute_active_set(
                    y_j[self.constraint_manager.v2d]
                )[active_idx]
            ):
                h = self.constraint_manager.evaluate_active(
                    coords_dof[self.constraint_manager.v2d], active_idx
                )
                if self.mode == "complete":
                    lambd = np.linalg.solve(A @ S_inv @ A.T, h)
                    y_j = y_j - S_inv @ A.T @ lambd
                else:
                    lambd = np.linalg.solve(A @ A.T, h)
                    y_j = y_j - A.T @ lambd

            else:
                return y_j

        print("Failed projection!")

    def compute_step(
        self,
        coords_dof,
        search_direction_dof,
        stepsize,
        active_idx,
        constraint_gradient,
        dropped_idx,
    ):
        def func(lambd):
            projected_step = self.project_to_working_set(
                coords_dof,
                search_direction_dof,
                lambd,
                active_idx,
                constraint_gradient,
            )
            if not projected_step is None:
                return np.max(
                    self.constraint_manager.evaluate(
                        projected_step[self.constraint_manager.v2d]
                    )[np.logical_and(~active_idx, ~dropped_idx)]
                )
            else:
                return 100.0

        while True:
            trial_step = self.project_to_working_set(
                coords_dof,
                search_direction_dof,
                stepsize,
                active_idx,
                constraint_gradient,
            )
            if trial_step is None:
                stepsize /= 2.0
            else:
                break

        if not np.all(
            self.constraint_manager.is_feasible(trial_step[self.constraint_manager.v2d])
        ):
            feasible_stepsize = optimize.root_scalar(
                func, bracket=(0.0, stepsize), xtol=1e-10
            ).root
            feasible_step = self.project_to_working_set(
                coords_dof,
                search_direction_dof,
                feasible_stepsize,
                active_idx,
                constraint_gradient,
            )

            assert np.all(
                self.constraint_manager.is_feasible(
                    feasible_step[self.constraint_manager.v2d]
                )
            )
            return feasible_step, feasible_stepsize
        else:
            return trial_step, stepsize

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
        self, search_direction: List[fenics.Function]
    ) -> None:
        """Restricts the search direction to the inactive set.

        Args:
            search_direction: The current search direction (will be overwritten).

        """
        pass
