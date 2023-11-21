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

from typing import TYPE_CHECKING

import fenics
import numpy as np
from petsc4py import PETSc
from scipy import sparse

from cashocs import _exceptions
from cashocs import _utils
from cashocs._optimization.optimization_algorithms import optimization_algorithm

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import line_search as ls


class ProjectedGradientDescent(optimization_algorithm.OptimizationAlgorithm):
    """Projected gradient descent method."""

    def __init__(
        self,
        db: database.Database,
        optimization_problem: _typing.OptimizationProblem,
        line_search: ls.LineSearch,
    ) -> None:
        """This class is an implementation of the projected gradient descent method.

        This is based on Rosen's projected gradient method and is used to enforce
        constraints on the mesh quality in a semi-discrete setting.

        Args:
            db: The database of the problem.
            optimization_problem: The underlying shape optimization problem to be
                solved.
            line_search: The line search used computing an update of the mesh.

        """
        super().__init__(db, optimization_problem, line_search)
        self.mesh = optimization_problem.mesh_handler.mesh
        self.constraint_manager = optimization_problem.constraint_manager
        self.dropped_idx = np.array([False] * self.constraint_manager.no_constraints)
        self.mode = self.config.get("MeshQualityConstraints", "mode")

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
            print(f"No. active constraints: {sum(self.active_idx)}", flush=True)

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
                self.active_idx, self.constraint_gradient
            )
            for i in range(len(self.db.function_db.gradient)):
                self.search_direction[i].vector().vec().aypx(0.0, p_dof.vector().vec())
                self.search_direction[i].vector().apply("")

    def _compute_projected_gradient(
        self, active_idx, constraint_gradient
    ) -> tuple[fenics.Function, bool, np.ndarray]:
        converged = False
        no_constraints = constraint_gradient.shape[0]
        dropped_idx_list = []
        undroppable_idx = []

        while True:
            A = self.constraint_manager.compute_active_gradient(
                active_idx, constraint_gradient
            )
            AT = A.copy().transpose()
            B = A.matMult(AT)

            if self.mode == "complete":
                S = self.optimization_problem.form_handler.scalar_product_matrix[:, :]
                S_inv = np.linalg.inv(S)
                lambd = np.linalg.solve(
                    A @ S_inv @ A.T, -A @ self.gradient[0].vector()[:]
                )
                p = -(self.gradient[0].vector()[:] + S_inv @ A.T @ lambd)
            else:
                b = A.createVecLeft()
                A.mult(-self.gradient[0].vector().vec(), b)
                ksp = PETSc.KSP().create(comm=self.constraint_manager.mesh.mpi_comm())
                options = {
                    "ksp_type": "cg",
                    "ksp_max_it": 1000,
                    "ksp_rtol": self.constraint_manager.constraint_tolerance / 1e2,
                    "ksp_atol": 1e-30,
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    # "ksp_monitor_true_residual": None,
                }

                ksp.setOperators(B)
                _utils.setup_petsc_options([ksp], [options])

                lambd = B.createVecRight()
                ksp.solve(b, lambd)

                if ksp.getConvergedReason() < 0:
                    raise _exceptions.NotConvergedError(
                        "Gradient projection", "The gradient projection failed."
                    )

                p = AT.createVecLeft()
                AT.mult(-lambd, p)
                p.axpy(-1.0, self.gradient[0].vector().vec())

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
            if np.linalg.norm(p) <= gamma and len(undroppable_idx) == 0:
                i_min_list = np.argsort(lambd_ineq)
                i_min_list_padded = np.where(self.constraint_manager.inequality_mask)[
                    0
                ][i_min_list]
                filter = ~np.in1d(i_min_list_padded, undroppable_idx)
                i_min_padded = i_min_list_padded[filter][0]

                active_idx[i_min_padded] = False
                dropped_idx_list.append(i_min_padded)
                # print(f"Dropped constraint {i_min_padded}")
                continue
            else:
                break

        p_dof = fenics.Function(self.db.function_db.control_spaces[0])
        p_dof.vector().set_local(p.getArray())
        p_dof.vector().apply("")

        if (
            np.sqrt(self.form_handler.scalar_product([p_dof], [p_dof]))
            <= self.atol + self.rtol * self.gradient_norm_initial
        ):
            lambda_min = np.min(lambd_ineq)

            if lambda_min >= 0.0:
                converged = True
                print("Algorithm converged!")
            else:
                pass

        dropped_idx = np.array([False] * len(self.active_idx))
        dropped_idx[dropped_idx_list] = True

        return p_dof, converged, dropped_idx

    def _compute_active_constraints(self) -> None:
        pass
