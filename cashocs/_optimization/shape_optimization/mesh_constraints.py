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

import abc

import fenics
import numpy as np
from scipy import sparse

from cashocs import _utils


class MeshConstraint(abc.ABC):
    type = None

    def __init__(self, mesh, deformation_space) -> None:
        self.mesh = mesh
        self.v2d = fenics.vertex_to_dof_map(deformation_space)
        self.d2v = fenics.dof_to_vertex_map(deformation_space)

    @abc.abstractmethod
    def evaluate(self, coords_sequential) -> None:
        pass

    @abc.abstractmethod
    def compute_gradient(self, coords_sequential) -> None:
        pass


class FixedBoundaryConstraint(MeshConstraint):
    type = "equality"

    def __init__(self, mesh, boundaries, config, deformation_space):
        super().__init__(mesh, deformation_space)
        self.boundaries = boundaries
        self.config = config

        self.dim = self.mesh.geometry().dim()
        self.mesh.init(self.mesh.topology().dim() - 1, 0)
        self.facets = np.array(
            self.mesh.topology()(self.mesh.topology().dim() - 1, 0)()
        ).reshape(-1, self.dim)

        shape_bdry_fix = self.config.getlist("ShapeGradient", "shape_bdry_fix")

        (
            self.fixed_idcs,
            self.fixed_coordinates,
        ) = self._compute_fixed_coordinate_indices(shape_bdry_fix)
        self.no_constraints = len(self.fixed_idcs)
        self.no_vertices = self.mesh.num_vertices()
        self.is_necessary = np.array([False] * self.no_constraints)
        self.fixed_gradient = self._compute_fixed_gradient()

    def _compute_fixed_coordinate_indices(
        self, shape_bdry_fix: list[int] | int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        bdry_fix_list = _utils.enlist(shape_bdry_fix)

        fixed_idcs = []
        for i in bdry_fix_list:
            idx_i = self.facets[self.boundaries.where_equal(i)]
            fixed_idcs += idx_i.reshape(-1).tolist()

        fixed_idcs = np.array(list(set(fixed_idcs)))
        fixed_idcs.sort()

        if self.dim == 2:
            fixed_idcs = np.array(
                [self.dim * fixed_idcs, self.dim * fixed_idcs + 1]
            ).T.reshape(-1)
        elif self.dim == 3:
            fixed_idcs = np.array(
                [
                    self.dim * fixed_idcs,
                    self.dim * fixed_idcs + 1,
                    self.dim * fixed_idcs + 2,
                ]
            ).T.reshape(-1)

        if len(fixed_idcs) > 0:
            fixed_coordinates = self.mesh.coordinates().copy().reshape(-1)[fixed_idcs]
        else:
            fixed_coordinates = None

        return fixed_idcs, fixed_coordinates

    def evaluate(self, coords_seq) -> np.ndarray:
        if len(self.fixed_idcs) > 0:
            return coords_seq[self.fixed_idcs] - self.fixed_coordinates
        else:
            return np.array([])

    def _compute_fixed_gradient(self) -> sparse.csr_matrix:
        if len(self.fixed_idcs) > 0:
            rows = np.arange(len(self.fixed_idcs))
            cols = self.v2d[self.fixed_idcs]
            vals = np.ones(len(self.fixed_idcs))
            csr = rows, cols, vals
            shape = (len(self.fixed_idcs), self.no_vertices * self.dim)
            gradient = sparse2scipy(csr, shape)
        else:
            shape = (0, self.no_vertices * self.dim)
            gradient = sparse2scipy(([], [], []), shape)
        return gradient

    def compute_gradient(self, coords_seq) -> sparse.csr_matrix:
        return self.fixed_gradient


class TriangleAngleConstraint(MeshConstraint):
    type = "inequality"

    def __init__(self, mesh, config, deformation_space):
        super().__init__(mesh, deformation_space)
        self.config = config

        self.dim = self.mesh.geometry().dim()
        self.min_angle = self.config.getfloat("MeshQualityConstraints", "min_angle")
        self.cells = self.mesh.cells()

        self.no_constraints = 3 * len(self.cells)
        self.is_necessary = np.array([True] * self.no_constraints)

    # ToDo: Parallel implementation,
    #  compute the minimum angle of each element directly and
    #  use this as threshold (scaled with a factor),
    #  c++ implementation,
    #  maybe return a list, so that appending is faster
    def evaluate(self, coords_seq) -> np.ndarray:
        values = []

        coords = coords_seq.reshape(-1, self.dim)

        for cell in self.cells:
            x_local = coords[cell]
            r_01 = x_local[0] - x_local[1]
            r_02 = x_local[0] - x_local[2]
            r_12 = x_local[1] - x_local[2]

            alpha = np.arccos(
                r_01.dot(r_02) / (np.linalg.norm(r_01) * np.linalg.norm(r_02))
            )
            beta = np.arccos(
                r_01.dot(-r_12) / (np.linalg.norm(r_01) * np.linalg.norm(r_12))
            )
            gamma = np.arccos(
                r_02.dot(r_12) / (np.linalg.norm(r_02) * np.linalg.norm(r_12))
            )

            values += [alpha, beta, gamma]

        values = np.array(values)
        values = self.min_angle * 2 * np.pi / 360.0 - values

        return values

    # ToDo: Parallel implementation,
    #  subtract the minimum angle directly,
    #  c++ implementation
    def compute_gradient(self, coords_seq) -> sparse.csr_matrix:
        coords = coords_seq.reshape(-1, self.dim)
        rows = []
        cols = []
        vals = []
        for idx, cell in enumerate(self.cells):
            x_local = coords[cell]
            r_01 = x_local[0] - x_local[1]
            r_02 = x_local[0] - x_local[2]
            r_12 = x_local[1] - x_local[2]

            r_01 = np.pad(r_01, (0, 1))
            r_02 = np.pad(r_02, (0, 1))
            r_12 = np.pad(r_12, (0, 1))

            tp_a1 = np.cross(-r_01, np.cross(-r_01, -r_02))
            tp_a2 = np.cross(-r_02, np.cross(-r_02, -r_01))
            tp_a1 /= np.linalg.norm(tp_a1)
            tp_a2 /= np.linalg.norm(tp_a2)

            tp_b0 = np.cross(r_01, np.cross(r_01, -r_12))
            tp_b2 = np.cross(-r_12, np.cross(-r_12, r_01))
            tp_b0 /= np.linalg.norm(tp_b0)
            tp_b2 /= np.linalg.norm(tp_b2)

            tp_c0 = np.cross(r_02, np.cross(r_02, r_12))
            tp_c1 = np.cross(r_12, np.cross(r_12, r_02))
            tp_c0 /= np.linalg.norm(tp_c0)
            tp_c1 /= np.linalg.norm(tp_c1)

            dad0 = (
                1.0 / np.linalg.norm(r_01) * tp_a1 + 1.0 / np.linalg.norm(r_02) * tp_a2
            )
            dad1 = -1.0 / np.linalg.norm(r_01) * tp_a1
            dad2 = -1.0 / np.linalg.norm(r_02) * tp_a2

            dbd0 = -1.0 / np.linalg.norm(r_01) * tp_b0
            dbd1 = (
                1.0 / np.linalg.norm(r_01) * tp_b0 + 1.0 / np.linalg.norm(r_12) * tp_b2
            )
            dbd2 = -1.0 / np.linalg.norm(r_12) * tp_b2

            dcd0 = -1.0 / np.linalg.norm(r_02) * tp_c0
            dcd1 = -1.0 / np.linalg.norm(r_12) * tp_c1
            dcd2 = (
                1.0 / np.linalg.norm(r_02) * tp_c0 + 1.0 / np.linalg.norm(r_12) * tp_c1
            )

            rows += [3 * idx] * 6
            cols += [
                2 * cell[0],
                2 * cell[0] + 1,
                2 * cell[1],
                2 * cell[1] + 1,
                2 * cell[2],
                2 * cell[2] + 1,
            ]
            vals += [dad0[0], dad0[1], dad1[0], dad1[1], dad2[0], dad2[1]]

            rows += [3 * idx + 1] * 6
            cols += [
                2 * cell[0],
                2 * cell[0] + 1,
                2 * cell[1],
                2 * cell[1] + 1,
                2 * cell[2],
                2 * cell[2] + 1,
            ]
            vals += [dbd0[0], dbd0[1], dbd1[0], dbd1[1], dbd2[0], dbd2[1]]

            rows += [3 * idx + 2] * 6
            cols += [
                2 * cell[0],
                2 * cell[0] + 1,
                2 * cell[1],
                2 * cell[1] + 1,
                2 * cell[2],
                2 * cell[2] + 1,
            ]
            vals += [dcd0[0], dcd0[1], dcd1[0], dcd1[1], dcd2[0], dcd2[1]]

        cols = self.v2d[cols]
        csr = rows, cols, vals
        shape = (int(3 * len(self.cells)), len(coords_seq))
        gradient = sparse2scipy(csr, shape)

        return gradient


class ConstraintManager:
    def __init__(self, config, mesh, boundaries, deformation_space):
        self.config = config
        self.mesh = mesh
        self.boundaries = boundaries
        self.has_constraints = False
        self.v2d = fenics.vertex_to_dof_map(deformation_space)
        self.d2v = fenics.dof_to_vertex_map(deformation_space)

        self.constraints = []
        self.constraint_tolerance = self.config.getfloat(
            "MeshQualityConstraints", "tol"
        )
        self.no_constraints = 0

        if self.config.getfloat("MeshQualityConstraints", "min_angle") > 0.0:
            self.constraints.append(
                FixedBoundaryConstraint(
                    self.mesh, self.boundaries, self.config, deformation_space
                )
            )
            self.constraints.append(
                TriangleAngleConstraint(self.mesh, self.config, deformation_space)
            )
            self.has_constraints = True

        if self.has_constraints:
            self.inequality_mask = self._compute_inequality_mask()
            self.no_constraints = len(self.inequality_mask)
            necessary_constraints = []
            for constraint in self.constraints:
                necessary_constraints.append(constraint.is_necessary)
            self.necessary_constraints = np.concatenate(necessary_constraints)

    def _compute_inequality_mask(self) -> np.ndarray:
        mask = []
        for constraint in self.constraints:
            if constraint.type == "equality":
                mask += [False] * constraint.no_constraints
            elif constraint.type == "inequality":
                mask += [True] * constraint.no_constraints

        return np.array(mask)

    def evaluate(self, coords_seq) -> np.ndarray:
        result = []
        for constraint in self.constraints:
            result.append(constraint.evaluate(coords_seq))

        if len(result) > 0:
            self.function_values = np.concatenate(result)
        else:
            return np.array([])

        return self.function_values

    def evaluate_active(self, coords_seq, active_idx) -> np.ndarray:
        function_values = self.evaluate(coords_seq)
        return function_values[active_idx]

    def compute_gradient(self, coords_seq) -> sparse.csr_matrix:
        result = []
        for constraint in self.constraints:
            result.append(constraint.compute_gradient(coords_seq))

        if len(result) > 0:
            self.gradient = sparse.vstack(result)
        else:
            return np.array([])

        return self.gradient

    def compute_active_gradient(
        self, active_idx, constraint_gradient: sparse.csr_matrix
    ) -> sparse.csr_matrix:
        return constraint_gradient[active_idx]

    def compute_active_set(self, coords_seq) -> np.ndarray:
        function_values = self.evaluate(coords_seq)
        result = np.abs(function_values) <= self.constraint_tolerance
        return result

    def is_necessary(self, active_idx: np.ndarray) -> bool:
        return np.any(np.logical_and(self.necessary_constraints, active_idx))

    def is_feasible(self, coords_seq) -> np.ndarray:
        function_values = self.evaluate(coords_seq)
        return np.array(
            [bool(value <= self.constraint_tolerance) for value in function_values]
        )


def sparse2scipy(csr, shape=None) -> sparse.csr_matrix:
    rows = csr[0]
    cols = csr[1]
    vals = csr[2]
    A = sparse.csr_matrix((vals, (rows, cols)), shape=shape)
    return A


def sparse2petsc(csr):
    pass
