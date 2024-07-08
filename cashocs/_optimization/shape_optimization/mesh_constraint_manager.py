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

"""Constraints for the mesh / shape in shape optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy import sparse

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils
from cashocs._optimization.shape_optimization import mesh_constraints
import cashocs.io

if TYPE_CHECKING:
    from cashocs._optimization import optimization_algorithms


class ConstraintManager:
    """This class manages all (selected) mesh quality constraints.

    It is used to treat arbitrary constraints (on the mesh coordinates) in a unified
    manner.
    """

    def __init__(
        self,
        config: cashocs.io.Config,
        mesh: fenics.Mesh,
        boundaries: fenics.MeshFunction,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the constraint manager.

        Args:
            config: The configuration of the optimization problem.
            mesh: The underlying FEM mesh.
            boundaries: The :py:class`fenics.MeshFunction` used to mark the boundaries
                of the mesh.
            deformation_space: A :py:class`fenics.FunctionSpace` of vector CG1 elements
                for mesh deformations.

        """
        self.config = config
        self.mesh = mesh
        self.boundaries = boundaries

        self.has_constraints = (
            self.config.getfloat("MeshQualityConstraints", "min_angle") > 0.0
        ) or (
            self.config.getfloat(
                "MeshQualityConstraints", "feasible_angle_reduction_factor"
            )
            > 0.0
        )
        self.v2d = fenics.vertex_to_dof_map(deformation_space)
        self.d2v = fenics.dof_to_vertex_map(deformation_space)
        loc0, loc1 = deformation_space.dofmap().ownership_range()
        self.d2v_local = self.d2v[: loc1 - loc0]

        fun = fenics.Function(deformation_space)
        self.local_petsc_size = fun.vector().vec().getSizes()[0]
        self.comm = self.mesh.mpi_comm()

        self.constraints: list[mesh_constraints.MeshConstraint] = []
        self.constraint_tolerance: float = self.config.getfloat(
            "MeshQualityConstraints", "tol"
        )
        self.no_constraints = 0

        if self.has_constraints:
            if self.mesh.geometry().dim() == 2:
                angle_constraint: mesh_constraints.AngleConstraint = (
                    mesh_constraints.TriangleAngleConstraint(
                        self.mesh, self.config, deformation_space
                    )
                )

            elif self.mesh.geometry().dim() == 3:
                angle_constraint = mesh_constraints.DihedralAngleConstraint(
                    self.mesh, self.config, deformation_space
                )
            else:
                raise _exceptions.InputError(
                    "ConstraintManager",
                    "mesh",
                    "Can only use mesh quality constraints for 2D and 3D meshes.",
                )

            self.constraints.append(angle_constraint)

            additional_fixed_idcs = angle_constraint.bad_idcs
            additional_fixed_coordinates = angle_constraint.bad_coordinates

            self.constraints.append(
                mesh_constraints.FixedVertexConstraint(
                    self.mesh,
                    self.boundaries,
                    self.config,
                    deformation_space,
                    additional_fixed_idcs=additional_fixed_idcs,
                    additional_fixed_coordinates=additional_fixed_coordinates,
                )
            )

            self.inequality_mask = self._compute_inequality_mask()
            self.no_constraints = len(self.inequality_mask)
            necessary_constraints = []
            for constraint in self.constraints:
                necessary_constraints.append(constraint.is_necessary)
            self.necessary_constraints = np.concatenate(necessary_constraints)

        if not np.all(self.is_feasible(self.mesh.coordinates().copy().reshape(-1))):
            raise _exceptions.InputError(
                "ShapeOptimizationProblem",
                "mesh",
                "You must supply a mesh which is feasible w.r.t. the mesh "
                "quality constraints, or try using a lower minimum angle.",
            )

    def _compute_inequality_mask(self) -> np.ndarray:
        """Computes the mask which distinguishes between (in-)equality constaints.

        Returns:
            A boolean array (a mask) which is `True` when the corresponding constraint
            is an inequality constraint.

        """
        mask = []
        for constraint in self.constraints:
            if constraint.type == "equality":
                mask += [False] * constraint.no_constraints
            elif constraint.type == "inequality":
                mask += [True] * constraint.no_constraints

        return np.array(mask)

    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        """Evaluates all (selected) mesh quality constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A numpy array with the values of the constraint functions.

        """
        result = []
        for constraint in self.constraints:
            result.append(constraint.evaluate(coords_seq))

        if len(result) > 0:
            self.function_values: np.ndarray = np.concatenate(result)
        else:
            return np.array([])

        return self.function_values

    def evaluate_active(
        self, coords_seq: np.ndarray, active_idx: np.ndarray
    ) -> np.ndarray:
        """Evaluates only the constraints corresponding to the working set.

        Args:
            coords_seq: The flattened list of mesh coordinates.
            active_idx: The mask of active indices corresponding to the working set.

        Returns:
            A numpy array containing only the values of the constraint functions that
            are in the working set.

        """
        function_values: np.ndarray = self.evaluate(coords_seq)[active_idx]
        return function_values

    def _compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        result = []
        for constraint in self.constraints:
            result.append(constraint.compute_gradient(coords_seq))

        if len(result) > 0:
            self.gradient = sparse.vstack(result)
        else:
            return np.array([])

        return self.gradient

    def compute_active_gradient(
        self, active_idx: np.ndarray, constraint_gradient: sparse.csr_matrix
    ) -> PETSc.Mat:
        """Computes the gradient of those constraint functions that are active.

        Args:
            active_idx: The boolean array corresponding to the current working set.
            constraint_gradient: The entire (sparse) constraint gradient.

        Returns:
            A sparse matrix of the constraint gradient w.r.t. the working set.

        """
        scipy_matrix = constraint_gradient[active_idx]

        petsc_matrix = _utils.linalg.scipy2petsc(
            scipy_matrix,
            self.mesh.mpi_comm(),
            local_size=self.local_petsc_size,
        )
        return petsc_matrix

    def compute_active_set(self, coords_seq: np.ndarray) -> np.ndarray:
        """Computes the working set.

        The working set is the set of constraints which are active at a given point.

        Args:
            coords_seq: The list of flattened mesh coordinates.

        Returns:
            A boolean array which is `True` in component `i` if constraint `i` is
            active.

        """
        function_values = self.evaluate(coords_seq)
        result: np.ndarray = np.abs(function_values) <= self.constraint_tolerance
        return result

    def is_necessary(self, active_idx: np.ndarray | None) -> bool:
        """Checks, whether constraint treatment is necessary based on the working set.

        This can be `False`, e.g., when only fixed boundary constraints are active (as
        these are directly treated in the gradient deformation) or no constraints are
        currently active.

        Args:
            active_idx: The boolean array representing the working set.

        Returns:
            A boolean, which is `True` when active treatment of the constraints is
            necessary and `False` otherwise.

        """
        if self.has_constraints:
            necessary_constraints_gathered = self.comm.allgather(
                self.necessary_constraints
            )
            necessary_constraints_global = np.concatenate(
                necessary_constraints_gathered
            )
            active_idx_gathered = self.comm.allgather(active_idx)
            active_idx_global = np.concatenate(active_idx_gathered)

            result = bool(
                np.any(np.logical_and(necessary_constraints_global, active_idx_global))
            )
            return result
        else:
            return False

    def is_feasible(self, coords_seq: np.ndarray) -> np.ndarray:
        """Checks, whether a given point is feasible w.r.t. all constraints.

        Args:
            coords_seq: The list of flattened mesh coordinates for the trial point.

        Returns:
            A boolean array, which is `True` in component `i` if the corresponding
            constraint is not violated, else `False`.

        """
        function_values = self.evaluate(coords_seq)
        function_values_list = self.comm.allgather(function_values)
        function_values_global = np.concatenate(function_values_list)

        result: np.ndarray = np.less_equal(
            function_values_global, self.constraint_tolerance
        )
        return result

    def compute_projected_search_direction(
        self,
        search_direction: list[fenics.Function],
        solver: optimization_algorithms.OptimizationAlgorithm,
    ) -> tuple[
        fenics.Function,
        bool,
        np.ndarray | None,
        sparse.csr_matrix | None,
        np.ndarray | None,
    ]:
        """Projects the search direction to the tangent space of the active constraints.

        Args:
            search_direction: The current search direction used in the solver.
            solver: The solver, which is used to solve the optimization problem.

        Returns:
            A tuple `(p_dof, converged, active_idx, constraint_gradient, dropped_idx)`,
            where `p_dof` is the projected search direction in dof-based ordering,
            `active_idx` is the index set of the active constraints (those in the
            current working set), `constraint_gradient` is the gradient matrix of the
            constraints, `dropped_idx` is the index set of all dropped constraints.

        """
        coords_sequential = self.mesh.coordinates().copy().reshape(-1)
        constraint_gradient = self._compute_gradient(coords_sequential)
        active_idx = self.compute_active_set(coords_sequential)
        self._compute_number_of_active_constraints(active_idx)

        if self.is_necessary(active_idx):
            return self._compute_projected_search_direction(
                search_direction, solver, active_idx, constraint_gradient
            )
        else:
            p_dof = fenics.Function(search_direction[0].function_space())
            p_dof.vector().vec().aypx(0.0, search_direction[0].vector().vec())
            p_dof.vector().apply("")
            if active_idx is not None:
                dropped_idx = np.array([False] * len(active_idx))
            else:
                dropped_idx = None

            return p_dof, False, active_idx, constraint_gradient, dropped_idx

    def _compute_projected_search_direction(
        self,
        search_direction: list[fenics.Function],
        solver: optimization_algorithms.OptimizationAlgorithm,
        active_idx: np.ndarray,
        constraint_gradient: sparse.csr_matrix,
    ) -> tuple[fenics.Function, bool, np.ndarray, sparse.csr_matrix, np.ndarray]:
        converged = False
        has_dropped_constraints = False
        has_undroppable_constraints = False
        no_constraints = constraint_gradient.shape[0]
        dropped_idx_list: list[int] = []
        undroppable_idx = []
        lambda_min_rank = 0

        while True:
            # pylint: disable=invalid-name
            A = self.compute_active_gradient(active_idx, constraint_gradient)
            AT = A.copy().transpose()  # pylint: disable=invalid-name
            B = A.matMult(AT)  # pylint: disable=invalid-name

            p, lambd = self._project_search_direction_to_tangent_space(
                search_direction, A, AT, B
            )

            if has_dropped_constraints:
                if not self._check_for_feasibility_of_dropped_constraints(
                    p, constraint_gradient, dropped_idx_list
                ):
                    if self.comm.rank == lambda_min_rank:
                        relevant_idx = dropped_idx_list.pop()
                        active_idx[relevant_idx] = True
                        undroppable_idx.append(relevant_idx)
                    self.comm.barrier()
                    has_undroppable_constraints = True
                    continue

            p_dof = fenics.Function(solver.db.function_db.control_spaces[0])
            p_dof.vector().set_local(p.getArray())
            p_dof.vector().apply("")
            solver.form_handler.apply_shape_bcs(p_dof)
            p_dof.vector().apply("")

            lambd_padded = np.zeros(no_constraints)

            lambd_padded[active_idx] = lambd
            lambd_ineq = lambd_padded[self.inequality_mask]

            lambda_min_list = self.comm.allgather(lambd_ineq.min())
            lambda_min_rank = int(np.argmin(lambda_min_list))
            lambda_min = np.min(lambda_min_list)
            gamma = -np.minimum(0.0, lambda_min)

            if (
                np.sqrt(solver.form_handler.scalar_product([p_dof], [p_dof])) <= gamma
                and not has_undroppable_constraints
            ):
                if self.comm.rank == lambda_min_rank:
                    i_min_list = np.argsort(lambd_ineq)
                    i_min_list_padded = np.where(self.inequality_mask)[0][i_min_list]
                    mask = ~np.isin(i_min_list_padded, undroppable_idx)
                    i_min_padded = i_min_list_padded[mask][0]

                    active_idx[i_min_padded] = False
                    dropped_idx_list.append(i_min_padded)
                    print(f"Dropped constraint {i_min_padded}", flush=True)

                has_dropped_constraints = True
                self.comm.barrier()
                continue
            else:
                break

        if (
            np.sqrt(solver.form_handler.scalar_product([p_dof], [p_dof]))
            <= solver.atol + solver.rtol * solver.gradient_norm_initial
        ):
            lambda_min = np.min(lambd_ineq)

            if lambda_min >= 0.0:
                converged = True
                print("Algorithm converged!")
            else:
                pass

        dropped_idx = np.array([False] * len(active_idx))
        dropped_idx[dropped_idx_list] = True

        return p_dof, converged, active_idx, constraint_gradient, dropped_idx

    def _project_search_direction_to_tangent_space(
        self,
        search_direction: list[fenics.Function],
        A: PETSc.Mat,  # pylint: disable=invalid-name
        AT: PETSc.Mat,  # pylint: disable=invalid-name
        B: PETSc.Mat,  # pylint: disable=invalid-name
    ) -> tuple[PETSc.Vec, PETSc.Vec]:
        # if self.mode == "complete":
        #    pylint: disable=invalid-name
        #    S = solver.form_handler.scalar_product_matrix[:, :]
        #    S_inv = np.linalg.inv(S)  # pylint: disable=invalid-name
        #    lambd = np.linalg.solve(A @ S_inv @ A.T, -A @ self.gradient[0].vector()[:])
        #    p = -(self.gradient[0].vector()[:] + S_inv @ A.T @ lambd)
        # else:

        b = A.createVecLeft()
        A.mult(search_direction[0].vector().vec(), b)
        ksp = PETSc.KSP().create(comm=self.mesh.mpi_comm())
        options: dict[str, float | int | str | None] = {
            "ksp_type": "cg",
            "ksp_max_it": 1000,
            "ksp_rtol": self.constraint_tolerance / 1e2,
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
        p.axpy(1.0, search_direction[0].vector().vec())

        return p, lambd

    def _check_for_feasibility_of_dropped_constraints(
        self,
        p: PETSc.Vec,
        constraint_gradient: sparse.csr_matrix,
        dropped_idx_list: list[int],
    ) -> bool:
        scipy_matrix = constraint_gradient[dropped_idx_list]
        petsc_matrix = _utils.linalg.scipy2petsc(
            scipy_matrix,
            self.comm,
            local_size=self.local_petsc_size,
        )
        res = petsc_matrix.getVecLeft()
        petsc_matrix.mult(p, res)
        directional_constraint_derivative_max = res.max()[1]
        return bool(directional_constraint_derivative_max < 1e-12)

    def _compute_number_of_active_constraints(self, active_idx: np.ndarray) -> None:
        no_active_equality_constraints_local = sum(active_idx[~self.inequality_mask])
        no_active_inequality_constraints_local = sum(active_idx[self.inequality_mask])

        no_active_equality_constraints_global = self.comm.allreduce(
            no_active_equality_constraints_local, op=MPI.SUM
        )
        no_active_inequality_constraints_global = self.comm.allreduce(
            no_active_inequality_constraints_local, op=MPI.SUM
        )

        _loggers.debug(
            f"{no_active_equality_constraints_global} equality "
            f"and {no_active_inequality_constraints_global} inequality constraints "
            "are currently active."
        )
