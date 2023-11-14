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

"""Projected Gradient Descent for PDE constrained optimization with constraints."""

from __future__ import annotations

import abc

import fenics
import numpy as np

from cashocs._optimization.optimization_algorithms import optimization_algorithm
from cashocs import _utils
from cashocs import _exceptions


class ProjectedGradientDescent(optimization_algorithm.OptimizationAlgorithm):
    """Projected gradient descent method."""

    def __init__(self, db, optimization_problem, line_search) -> None:
        super().__init__(db, optimization_problem, line_search)
        self.mesh = optimization_problem.mesh_handler.mesh
        self.constraint_manager = optimization_problem.constraint_manager
        self.dropped_idx = np.array([False] * self.constraint_manager.no_constraints)

    def run(self) -> None:
        """Performs the optimization with the gradient descent method."""
        while True:
            self.coords_sequential = self.mesh.coordinates().copy().reshape(-1)
            constraint_values = self.constraint_manager.evaluate(self.coords_sequential)
            if self.iteration == 0 and constraint_values.max() > 0.0:
                raise _exceptions.InputError(
                    "ShapeOptimizationProblem",
                    "mesh",
                    "You must supply a mesh which is feasible w.r.t. the mesh "
                    "quality constraints, or try using a lower minimum angle.",
                )

            self.constraint_gradient = self.constraint_manager.compute_gradient(
                self.coords_sequential
            )
            self.active_idx = self.constraint_manager.compute_active_set(
                self.coords_sequential
            )
            print(f"No. active constraints: {sum(self.active_idx)}")

            self.compute_gradient()
            self.gradient_norm = self.compute_gradient_norm()

            if self.convergence_test():
                break

            self.evaluate_cost_functional()

            self.compute_search_direction()
            self.line_search.perform(
                self,
                self.search_direction,
                self.has_curvature_info,
                self.active_idx,
                self.constraint_gradient,
                self.dropped_idx,
            )

            self.iteration += 1
            if self.nonconvergence():
                break

    def compute_search_direction(self) -> None:
        """Computes the search direction for the projected gradient descent method."""
        if not self.constraint_manager.is_necessary(self.active_idx):
            for i in range(len(self.db.function_db.gradient)):
                self.search_direction[i].vector().vec().aypx(
                    0.0, -self.db.function_db.gradient[i].vector().vec()
                )
                self.search_direction[i].vector().apply("")
        else:
            p_dof, converged, self.dropped_idx = self._compute_projected_gradient(
                self.coords_sequential, self.active_idx, self.constraint_gradient
            )
            for i in range(len(self.db.function_db.gradient)):
                self.search_direction[i].vector().vec().aypx(0.0, p_dof.vector().vec())
                self.search_direction[i].vector().apply("")

    def _compute_projected_gradient(
        self, coords_sequential, active_idx, constraint_gradient
    ) -> tuple[fenics.Function, bool, np.ndarray]:
        converged = False
        no_constraints = len(constraint_gradient)
        dropped_idx_list = []
        undroppable_idx = []

        while True:
            A = self.constraint_manager.compute_active_gradient(
                active_idx, constraint_gradient
            )

            lambd = np.linalg.solve(A @ A.T, -A @ self.gradient[0].vector()[:])
            p = -(self.gradient[0].vector()[:] + A.T @ lambd)

            if len(dropped_idx_list) > 0:
                if not np.all(constraint_gradient[dropped_idx_list] @ p < 1e-12):
                    relevant_idx = dropped_idx_list.pop()
                    active_idx[relevant_idx] = True
                    undroppable_idx.append(relevant_idx)
                    continue

            lambd_padded = np.zeros(no_constraints)
            lambd_padded[active_idx] = lambd
            lambd_ineq = lambd_padded[self.constraint_manager.inequality_mask]

            gamma = -np.minimum(0.0, lambd_ineq.min())
            if np.linalg.norm(p) <= gamma and not len(undroppable_idx) > 0:
                i_min_list = np.argsort(lambd_ineq)
                i_min_list_padded = np.where(self.constraint_manager.inequality_mask)[
                    0
                ][i_min_list]
                filter = ~np.in1d(i_min_list_padded, undroppable_idx)
                i_min_padded = i_min_list_padded[filter][0]

                active_idx[i_min_padded] = False
                dropped_idx_list.append(i_min_padded)
                print(f"Dropped constraint {i_min_padded}")
                continue
            else:
                break

        print(f"{lambd_ineq.min() = :.3e}  {np.linalg.norm(p) = :.3e}")

        p_dof = fenics.Function(self.db.function_db.control_spaces[0])
        p_dof.vector()[:] = p

        if (
            np.sqrt(self.form_handler.scalar_product([p_dof], [p_dof]))
            <= self.atol + self.rtol * self.gradient_norm_initial
        ):
            lambda_min = np.min(lambd_ineq)

            if lambda_min >= 0.0:
                converged = True
            else:
                pass

        dropped_idx = np.array([False] * len(self.active_idx))
        dropped_idx[dropped_idx_list] = True

        return p_dof, converged, dropped_idx

    def _compute_active_constraints(self) -> None:
        pass
