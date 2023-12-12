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
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy import optimize
from scipy import sparse

from cashocs import _exceptions
from cashocs import _forms
from cashocs import _loggers
from cashocs import _utils
from cashocs._optimization import optimization_variable_abstractions

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import shape_optimization
    from cashocs._optimization.shape_optimization import mesh_constraints


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
            constraint_manager: The constraint manager for mesh quality constraints.

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
        self,
        search_direction: List[fenics.Function],
        stepsize: float,
        beta: float,
        active_idx: np.ndarray | None = None,
        constraint_gradient: np.ndarray | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> float:
        """Updates the optimization variables based on a line search.

        This variant is used when constraints are present and projects the step back
        to the surface of the constraints in the active set.

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
        if (
            active_idx is not None
            and constraint_gradient is not None
            and dropped_idx is not None
        ):
            return self._update_constrained_optimization_variables(
                search_direction,
                stepsize,
                beta,
                active_idx,
                constraint_gradient,
                dropped_idx,
            )
        else:
            return self._update_optimization_variables(search_direction, stepsize, beta)

    def _update_optimization_variables(
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

    def _update_constrained_optimization_variables(
        self,
        search_direction: List[fenics.Function],
        stepsize: float,
        beta: float,
        active_idx: np.ndarray,
        constraint_gradient: np.ndarray,
        dropped_idx: np.ndarray,
    ) -> float:
        """Updates the optimization variables based on a line search.

        This variant is used when constraints are present and projects the step back
        to the surface of the constraints in the active set.

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
            coords_sequential = self.mesh_handler.mesh.coordinates().copy().reshape(-1)
            coords_dof = coords_sequential[self.constraint_manager.d2v]
            search_direction_vertex = (
                self.mesh_handler.deformation_handler.dof_to_coordinate(
                    search_direction[0]
                )
            )
            search_direction_dof = search_direction_vertex.reshape(-1)[
                self.constraint_manager.d2v
            ]

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
                dof_deformation.vector().set_local(dof_deformation_vector)
                dof_deformation.vector().apply("")

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
                    break
            else:
                stepsize /= beta

        return stepsize

    def compute_step(
        self,
        coords_dof: np.ndarray,
        search_direction_dof: np.ndarray,
        stepsize: float,
        active_idx: np.ndarray,
        constraint_gradient: sparse.csr_matrix,
        dropped_idx: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Computes a feasible mesh movement subject to mesh quality constraints.

        Args:
            coords_dof: The current coordinates, ordered in a dof-based way.
            search_direction_dof: The search direction, given also in a dof-based way.
            stepsize: The trial size of the step.
            active_idx: A boolean mask used to identify the constraints that are
                currently in the working set.
            constraint_gradient: The sparse matrix containing (all) gradients of the
                constraints.
            dropped_idx: A boolean mask of indices of constraints, which have been
                dropped from the working set to generate larger descent.

        Returns:
            A tuple `feasible_step, feasible_stepsize`, where `feasible_step` is a
            feasible mesh configuration (based on all constraints) and
            `feasible_stepsize` is the corresponding stepsize taken.

        """
        comm = self.mesh_handler.mesh.mpi_comm()

        def func(lambd: float) -> float:
            projected_step = self.project_to_working_set(
                coords_dof,
                search_direction_dof,
                lambd,
                active_idx,
                constraint_gradient,
            )
            if projected_step is not None:
                rval = np.max(
                    self.constraint_manager.evaluate(
                        projected_step[self.constraint_manager.v2d]
                    )[np.logical_and(~active_idx, ~dropped_idx)]
                )
                value: float = comm.allreduce(rval, op=MPI.MAX)

                return value
            else:
                return 100.0

        while True:
            self.constraint_manager.comm.barrier()
            _loggers.debug(f"Doing a trial step with the stepsize {stepsize:.3e}")
            trial_step = self.project_to_working_set(
                coords_dof,
                search_direction_dof,
                stepsize,
                active_idx,
                constraint_gradient,
            )
            self.constraint_manager.comm.barrier()
            if trial_step is None:
                stepsize /= 2.0
            else:
                _loggers.debug(
                    "Found a stepsize which is feasible for the "
                    "currently active constraints in the working set."
                )
                break

        if not np.all(
            self.constraint_manager.is_feasible(trial_step[self.constraint_manager.v2d])
        ):
            _loggers.debug(
                "The step violates some new constraints. "
                "Computing the maximum stepsize until the a new constraint is active."
            )
            feasible_stepsize_local = optimize.root_scalar(
                func, bracket=(0.0, stepsize), xtol=1e-10
            ).root

            feasible_stepsize = comm.allreduce(feasible_stepsize_local, op=MPI.MIN)
            _loggers.debug(
                f"Stepsize until constraint boundary: {feasible_stepsize:.6e}"
            )
            feasible_step = self.project_to_working_set(
                coords_dof,
                search_direction_dof,
                feasible_stepsize,
                active_idx,
                constraint_gradient,
            )
            assert feasible_step is not None  # nosec B101

            assert np.all(  # nosec B101
                self.constraint_manager.is_feasible(
                    feasible_step[self.constraint_manager.v2d]
                )
            )
            return feasible_step, feasible_stepsize
        else:
            return trial_step, stepsize

    def project_to_working_set(
        self,
        coords_dof: np.ndarray,
        search_direction_dof: np.ndarray,
        stepsize: float,
        active_idx: np.ndarray,
        constraint_gradient: sparse.csr_matrix,
    ) -> np.ndarray | None:
        """Projects an (attempted) step back to the working set of active constraints.

        The step is of the form: `coords_dof + stepsize * search_direction_dof`, the
        working set is defined by `active_idx` and the gradient of (all) constraints is
        given in `constraint_gradient`.

        Args:
            coords_dof: The current coordinates, ordered in a dof-based way.
            search_direction_dof: The search direction, given also in a dof-based way.
            stepsize: The trial size of the step.
            active_idx: A boolean mask used to identify the constraints that are
                currently in the working set.
            constraint_gradient: The sparse matrix containing (all) gradients of the
                constraints.

        Returns:
            The projected step (if the projection was successful) or `None` otherwise.

        """
        comm = self.mesh_handler.mesh.mpi_comm()

        y_j: np.ndarray = coords_dof + stepsize * search_direction_dof
        # pylint: disable=invalid-name
        A = self.constraint_manager.compute_active_gradient(
            active_idx, constraint_gradient
        )
        AT = A.copy().transpose()  # pylint: disable=invalid-name
        B = A.matMult(AT)  # pylint: disable=invalid-name

        # if self.mode == "complete":
        #   S = self.form_handler.scalar_product_matrix[  # pylint: disable=invalid-name
        #       :, :
        #   ]
        #   S_inv = np.linalg.inv(S)  # pylint: disable=invalid-name

        for its in range(10):
            trial_active_idx = self.constraint_manager.compute_active_set(
                y_j[self.constraint_manager.v2d]
            )[active_idx]
            satisfies_previous_constraints_local = np.all(trial_active_idx)
            satisfies_previous_constraints = comm.allgather(
                satisfies_previous_constraints_local
            )
            satisfies_previous_constraints = np.all(satisfies_previous_constraints)

            h = self.constraint_manager.evaluate_active(
                y_j[self.constraint_manager.v2d], active_idx
            )
            residual = comm.allreduce(np.max(np.abs(h)), op=MPI.MAX)
            _loggers.debug(
                "Projection to the working set. "
                f"Iteration: {its}  Residual: {residual:.3e}"
            )

            if not satisfies_previous_constraints:
                # if self.mode == "complete":
                #     lambd = np.linalg.solve(A @ S_inv @ A.T, h)
                #     y_j = y_j - S_inv @ A.T @ lambd
                # else:
                ksp = PETSc.KSP().create(comm=self.constraint_manager.mesh.mpi_comm())

                options: dict[str, int | float | str | None] = {
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
                h_petsc = B.createVecLeft()
                h_petsc.array_w = h
                h_petsc.assemble()

                ksp.solve(h_petsc, lambd)

                if ksp.getConvergedReason() < 0:
                    raise _exceptions.NotConvergedError(
                        "Gradient projection", "The gradient projection failed."
                    )

                y_petsc = AT.createVecLeft()
                AT.mult(lambd, y_petsc)

                update = fenics.Function(self.db.function_db.control_spaces[0])
                update.vector().vec().aypx(0.0, y_petsc)
                update.vector().apply("")
                self.form_handler.apply_shape_bcs(update)
                update.vector().apply("")

                update_vertex = self.mesh_handler.deformation_handler.dof_to_coordinate(
                    update
                )
                update_dof = update_vertex.reshape(-1)[self.constraint_manager.d2v]
                y_j = y_j - update_dof

            else:
                _loggers.debug(
                    f"Projection to the working set successful after {its} iterations."
                )
                return y_j

        _loggers.debug("Back-Projection failed.")
        return None

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
