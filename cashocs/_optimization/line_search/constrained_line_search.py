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

"""Armijo line search algorithm."""

from __future__ import annotations

from typing import cast, List, Optional, Tuple

import fenics
from petsc4py import PETSc
from typing_extensions import TYPE_CHECKING

from cashocs._optimization.line_search import armijo_line_search

if TYPE_CHECKING:
    from cashocs._optimization import optimization_algorithms


class ConstrainedLineSearch(armijo_line_search.ArmijoLineSearch):
    """Implementation of the Armijo line search procedure."""

    def search(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
        active_idx,
        constraint_gradient,
        dropped_idx,
    ) -> Tuple[Optional[fenics.Function], bool]:
        """Performs the line search.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates whether the direction is
                (presumably) scaled.

        Returns:
            A tuple (defo, is_remeshed), where defo is accepted deformation / update
            or None, in case the update was not successful and is_remeshed is a boolean
            indicating whether a remeshing has been performed in the line search.

        """
        self.initialize_stepsize(solver, search_direction, has_curvature_info)
        is_remeshed = False

        opt_var_abstr = self.optimization_variable_abstractions

        while True:
            if self._check_for_nonconvergence(solver):
                return (None, False)

            if self.problem_type == "shape":
                self.decrease_measure_w_o_step = opt_var_abstr.compute_decrease_measure(
                    search_direction
                )
            self.stepsize = opt_var_abstr.update_constrained_optimization_variables(
                search_direction,
                self.stepsize,
                self.beta_armijo,
                active_idx,
                constraint_gradient,
                dropped_idx,
            )

            current_function_value = solver.objective_value
            objective_step = self._compute_objective_at_new_iterate(
                current_function_value
            )

            decrease_measure = self._compute_decrease_measure(search_direction)

            if self._satisfies_armijo_condition(
                objective_step, current_function_value, decrease_measure
            ):
                if self.optimization_variable_abstractions.requires_remeshing():
                    is_remeshed = (
                        self.optimization_variable_abstractions.mesh_handler.remesh(
                            solver
                        )
                    )
                    break

                if solver.iteration == 0:
                    self.armijo_stepsize_initial = self.stepsize
                break

            else:
                self.stepsize /= self.beta_armijo
                self.optimization_variable_abstractions.revert_variable_update()

        solver.stepsize = self.stepsize

        if not has_curvature_info:
            self.stepsize *= self.beta_armijo

        if self.problem_type == "shape":
            return (self.optimization_variable_abstractions.deformation, is_remeshed)
        else:
            return (None, False)

    def perform(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: List[fenics.Function],
        has_curvature_info: bool,
        active_idx,
        constraint_gradient,
        dropped_idx,
    ) -> None:
        """Performs a line search for the new iterate.

        Notes:
            This is the function that should be called in the optimization algorithm,
            it consists of a call to ``self.search`` and ``self.post_line_search``
            afterwards.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates, whether the search direction
                is (presumably) scaled.

        """
        deformation, is_remeshed = self.search(
            solver,
            search_direction,
            has_curvature_info,
            active_idx,
            constraint_gradient,
            dropped_idx,
        )
        if deformation is not None and self.config.getboolean(
            "ShapeGradient", "global_deformation"
        ):
            x = fenics.as_backend_type(deformation.vector()).vec()

            if not is_remeshed:
                transfer_matrix = solver.db.geometry_db.transfer_matrix
            else:
                transfer_matrix = solver.db.geometry_db.old_transfer_matrix

            transfer_matrix = cast(PETSc.Mat, transfer_matrix)

            _, temp = transfer_matrix.getVecs()
            transfer_matrix.mult(x, temp)
            self.global_deformation_vector.axpy(1.0, temp)
            self.deformation_function.vector().apply("")

        self.post_line_search()
