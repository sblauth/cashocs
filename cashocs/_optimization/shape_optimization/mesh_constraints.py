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
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy import sparse

from cashocs import _utils
import cashocs.io

cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <tuple>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;

void angles_triangle(const Cell& cell, std::vector<double>& angs)
{
    const Mesh& mesh = cell.mesh();
    angs.resize(3);
    const std::size_t i0 = cell.entities(0)[0];
    const std::size_t i1 = cell.entities(0)[1];
    const std::size_t i2 = cell.entities(0)[2];

    const Point p0 = Vertex(mesh, i0).point();
    const Point p1 = Vertex(mesh, i1).point();
    const Point p2 = Vertex(mesh, i2).point();
    Point e0 = p1 - p0;
    Point e1 = p2 - p0;
    Point e2 = p2 - p1;

    e0 /= e0.norm();
    e1 /= e1.norm();
    e2 /= e2.norm();

    angs[0] = acos(e0.dot(e1));
    angs[1] = acos(-e0.dot(e2));
    angs[2] = acos(e1.dot(e2));
}

py::array_t<double>
triangle_angles(std::shared_ptr<const Mesh> mesh)
{
    size_t idx = 0;
    auto n = mesh->num_cells();
    py::array_t<double> angles(3*n);
    auto buf = angles.request();
    double *ptr = (double *) buf.ptr;

    std::vector<double> angs;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        angles_triangle(*cell, angs);
        ptr[3*idx] = angs[0];
        ptr[3*idx+1] = angs[1];
        ptr[3*idx+2] = angs[2];
        idx += 1;
    }
    return angles;
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>>
triangle_angle_gradient(std::shared_ptr<const Mesh> mesh)
{
    size_t idx = 0;
    auto n = mesh->num_cells();

    py::array_t<int> rows(18*n);
    auto buf_rows = rows.request();
    int *ptr_rows = (int *) buf_rows.ptr;

    py::array_t<int> cols(18*n);
    auto buf_cols = cols.request();
    int *ptr_cols = (int *) buf_cols.ptr;

    py::array_t<double> vals(18*n);
    auto buf_vals = vals.request();
    double *ptr_vals = (double *) buf_vals.ptr;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        const std::size_t i0 = cell->entities(0)[0];
        const std::size_t i1 = cell->entities(0)[1];
        const std::size_t i2 = cell->entities(0)[2];

        const Point p0 = Vertex(*mesh, i0).point();
        const Point p1 = Vertex(*mesh, i1).point();
        const Point p2 = Vertex(*mesh, i2).point();
        Point e0 = p1 - p0;
        Point e1 = p2 - p0;
        Point e2 = p2 - p1;

        Point tpa1 = e0.cross(e0.cross(e1));
        Point tpa2 = e1.cross(e1.cross(e0));
        tpa1 /= tpa1.norm();
        tpa2 /= tpa2.norm();

        Point tpb0 = e0.cross(e0.cross(e2));
        Point tpb2 = e2.cross(e2.cross(-e0));
        tpb0 /= tpb0.norm();
        tpb2 /= tpb2.norm();

        Point tpc0 = e1.cross(e1.cross(-e2));
        Point tpc1 = e2.cross(e2.cross(-e1));
        tpc0 /= tpc0.norm();
        tpc1 /= tpc1.norm();

        Point dad0 = 1.0 / e0.norm() * tpa1 + 1.0 / e1.norm() * tpa2;
        Point dad1 = -1.0 / e0.norm() * tpa1;
        Point dad2 = -1.0 / e1.norm() * tpa2;

        Point dbd0 = -1.0 / e0.norm() * tpb0;
        Point dbd1 = 1.0 / e0.norm() * tpb0 + 1.0 / e2.norm() * tpb2;
        Point dbd2 = -1.0 / e2.norm() * tpb2;

        Point dcd0 = -1.0 / e1.norm() * tpc0;
        Point dcd1 = -1.0 / e2.norm() * tpc1;
        Point dcd2 = 1.0 / e1.norm() * tpc0 + 1.0 / e2.norm() * tpc1;

        ptr_rows[18*idx + 0] = 3*idx;
        ptr_rows[18*idx + 1] = 3*idx;
        ptr_rows[18*idx + 2] = 3*idx;
        ptr_rows[18*idx + 3] = 3*idx;
        ptr_rows[18*idx + 4] = 3*idx;
        ptr_rows[18*idx + 5] = 3*idx;

        ptr_cols[18*idx + 0] = 2 * i0 + 0;
        ptr_cols[18*idx + 1] = 2 * i0 + 1;
        ptr_cols[18*idx + 2] = 2 * i1 + 0;
        ptr_cols[18*idx + 3] = 2 * i1 + 1;
        ptr_cols[18*idx + 4] = 2 * i2 + 0;
        ptr_cols[18*idx + 5] = 2 * i2 + 1;

        ptr_vals[18*idx + 0] = dad0[0];
        ptr_vals[18*idx + 1] = dad0[1];
        ptr_vals[18*idx + 2] = dad1[0];
        ptr_vals[18*idx + 3] = dad1[1];
        ptr_vals[18*idx + 4] = dad2[0];
        ptr_vals[18*idx + 5] = dad2[1];

        ptr_rows[18*idx + 6] = 3*idx + 1;
        ptr_rows[18*idx + 7] = 3*idx + 1;
        ptr_rows[18*idx + 8] = 3*idx + 1;
        ptr_rows[18*idx + 9] = 3*idx + 1;
        ptr_rows[18*idx + 10] = 3*idx + 1;
        ptr_rows[18*idx + 11] = 3*idx + 1;

        ptr_cols[18*idx + 6] = 2 * i0 + 0;
        ptr_cols[18*idx + 7] = 2 * i0 + 1;
        ptr_cols[18*idx + 8] = 2 * i1 + 0;
        ptr_cols[18*idx + 9] = 2 * i1 + 1;
        ptr_cols[18*idx + 10] = 2 * i2 + 0;
        ptr_cols[18*idx + 11] = 2 * i2 + 1;

        ptr_vals[18*idx + 6] = dbd0[0];
        ptr_vals[18*idx + 7] = dbd0[1];
        ptr_vals[18*idx + 8] = dbd1[0];
        ptr_vals[18*idx + 9] = dbd1[1];
        ptr_vals[18*idx + 10] = dbd2[0];
        ptr_vals[18*idx + 11] = dbd2[1];

        ptr_rows[18*idx + 12] = 3*idx + 2;
        ptr_rows[18*idx + 13] = 3*idx + 2;
        ptr_rows[18*idx + 14] = 3*idx + 2;
        ptr_rows[18*idx + 15] = 3*idx + 2;
        ptr_rows[18*idx + 16] = 3*idx + 2;
        ptr_rows[18*idx + 17] = 3*idx + 2;

        ptr_cols[18*idx + 12] = 2 * i0 + 0;
        ptr_cols[18*idx + 13] = 2 * i0 + 1;
        ptr_cols[18*idx + 14] = 2 * i1 + 0;
        ptr_cols[18*idx + 15] = 2 * i1 + 1;
        ptr_cols[18*idx + 16] = 2 * i2 + 0;
        ptr_cols[18*idx + 17] = 2 * i2 + 1;

        ptr_vals[18*idx + 12] = dcd0[0];
        ptr_vals[18*idx + 13] = dcd0[1];
        ptr_vals[18*idx + 14] = dcd1[0];
        ptr_vals[18*idx + 15] = dcd1[1];
        ptr_vals[18*idx + 16] = dcd2[0];
        ptr_vals[18*idx + 17] = dcd2[1];

        idx += 1;
    }
    return std::make_tuple(rows, cols, vals);
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("triangle_angles", &triangle_angles);
    m.def("triangle_angle_gradient", &triangle_angle_gradient);
}
"""
mesh_quality = fenics.compile_cpp_code(cpp_code)


class MeshConstraint(abc.ABC):
    """This class represents a generic constraint on the mesh quality."""

    type = None

    def __init__(
        self, mesh: fenics.Mesh, deformation_space: fenics.FunctionSpace
    ) -> None:
        """Abstract base class representing an interface for mesh quality constraints.

        Args:
            mesh: The underlying finite element mesh, which is subjected to the mesh
            quality constraints.
            deformation_space: A vector CG1 FEM function space used for defining
                deformations of the mesh via perturbation of identity.

        """
        self.mesh = mesh
        self.dim = self.mesh.geometry().dim()

        self.v2d = fenics.vertex_to_dof_map(deformation_space)
        self.d2v = fenics.dof_to_vertex_map(deformation_space)

        dof_ownership_range = deformation_space.dofmap().ownership_range()
        self.local_offset = dof_ownership_range[0]
        self.no_vertices = self.mesh.num_entities_global(0)
        self.l2g_dofs = deformation_space.dofmap().tabulate_local_to_global_dofs()

        self._compute_parallel_relations()

    def _compute_parallel_relations(self) -> None:
        self.global_vertex_indices = self.mesh.topology().global_indices(0)
        V = fenics.FunctionSpace(self.mesh, "CG", 1)
        loc0, loc1 = V.dofmap().ownership_range()
        d2v = fenics.dof_to_vertex_map(V)
        self.global_vertex_indices_owned = self.global_vertex_indices[
            d2v[: loc1 - loc0]
        ]

        if self.dim == 2:
            self.cols_owned = np.array(
                [
                    self.dim * self.global_vertex_indices_owned,
                    self.dim * self.global_vertex_indices_owned + 1,
                ]
            ).T.reshape(-1)
        elif self.dim == 3:
            self.cols_owned = np.array(
                [
                    self.dim * self.global_vertex_indices_owned,
                    self.dim * self.global_vertex_indices_owned + 1,
                    self.dim * self.global_vertex_indices_owned + 2,
                ]
            ).T.reshape(-1)

        self.cols_owned.sort()

    @abc.abstractmethod
    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) \leq 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        pass

    @abc.abstractmethod
    def compute_gradient(self, coords_sequ: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        pass


class FixedBoundaryConstraint(MeshConstraint):
    """A discrete mesh constraint for fixed boundaries in shape optimization.

    This constraint is needed to incorporate other mesh constraints for the case that
    some part of the boundary is fixed during optimization. By default, this is already
    incorporated into the gradient deformation, so that this constraint alone is
    not necessary, however, it becomes so when other constraints are considered.
    The constraint is linear, so that the gradient only has to be computed once.
    """

    type = "equality"

    def __init__(
        self,
        mesh: fenics.Mesh,
        boundaries: fenics.MeshFunction,
        config: cashocs.io.Config,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the fixed boundary mesh constraint.

        Args:
            mesh: The corresponding finite element mesh.
            boundaries: The :py:class`fenics.MeshFunction` which is used to mark the
                boundaries of the mesh.
            config: The configuration of the optimization problem.
            deformation_space: The space of vector CG1 elements for mesh deformations.

        """
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
        self.is_necessary = np.array([False] * self.no_constraints)
        self.fixed_gradient = self._compute_fixed_gradient()

    def _compute_fixed_coordinate_indices(
        self, shape_bdry_fix: list[int] | int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Computes the (flattened) indices of the mesh coordinates, which are fixed.

        Args:
            shape_bdry_fix: An index or list of indices of the boundaries, which are
                fixed during shape optimization. This is specified in the configuration
                of the problem.

        Returns:
            A tuple `fixed_indices, fixed_coordinates`, where `fixed_indices` is an
            array containing the (flattened) indices of the fixed coordinates and
            `fixed_coordinates` contains the corresponding values of the fixed vertices.

        """
        bdry_fix_list = _utils.enlist(shape_bdry_fix)

        fixed_idcs = []
        for i in bdry_fix_list:
            idx_i = self.facets[self.boundaries.where_equal(i)]
            fixed_idcs += idx_i.reshape(-1).tolist()

        fixed_idcs = np.unique(fixed_idcs)

        if len(fixed_idcs) > 0:
            fixed_idcs_global = self.global_vertex_indices[fixed_idcs]
        else:
            fixed_idcs_global = []
        mask = np.isin(fixed_idcs_global, self.global_vertex_indices_owned)
        fixed_idcs_local_owned = fixed_idcs[mask]

        if self.dim == 2:
            fixed_idcs_local = np.array(
                [
                    self.dim * fixed_idcs_local_owned,
                    self.dim * fixed_idcs_local_owned + 1,
                ]
            ).T.reshape(-1)
        elif self.dim == 3:
            fixed_idcs_local = np.array(
                [
                    self.dim * fixed_idcs_local_owned,
                    self.dim * fixed_idcs_local_owned + 1,
                    self.dim * fixed_idcs_local_owned + 2,
                ]
            ).T.reshape(-1)

        if len(fixed_idcs_local) > 0:
            fixed_coordinates = (
                self.mesh.coordinates().copy().reshape(-1)[fixed_idcs_local]
            )
        else:
            fixed_coordinates = None

        return fixed_idcs_local, fixed_coordinates

    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) = 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`. The constraint returns the mesh coordinates corresponding
        to the boundaries that shall remain fixed during shape optimization.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        if len(self.fixed_idcs) > 0:
            return coords_seq[self.fixed_idcs] - self.fixed_coordinates
        else:
            return np.array([])

    def _compute_fixed_gradient(self) -> sparse.csr_matrix:
        """Computes the gradient of the constraint function.

        As the constraint is linear, the gradient is constant and, so, it is precomputed
        with this method. Also, the gradient is independent of the actual values of the
        mesh coordinates.

        Returns:
            A sparse matrix representation of the constraint gradient.

        """
        if len(self.fixed_idcs) > 0:
            rows = np.arange(len(self.fixed_idcs))

            cols = self.l2g_dofs[self.v2d[self.fixed_idcs]]
            vals = np.ones(len(self.fixed_idcs))
            csr = rows, cols, vals
            shape = (len(self.fixed_idcs), self.no_vertices * self.dim)
            gradient = sparse2scipy(csr, shape)
        else:
            shape = (0, self.no_vertices * self.dim)
            gradient = sparse2scipy(([], [], []), shape)
        return gradient

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        return self.fixed_gradient


class TriangleAngleConstraint(MeshConstraint):
    r"""A mesh quality constraint for the angles of triangles in the mesh.

    This ensures that the all angles :math:`\alpha` of the FEM mesh satisfy the
    constraint :math:`\alpha \geq \alpha_{min}`, which is equivalently rewritten as
    :math:`g(x) \leq 0` with :math:`g(x) = \alpha - \alpha_{min}`, where :math:`x` are
    the mesh coordinates.
    """
    type = "inequality"

    def __init__(
        self,
        mesh: fenics.Mesh,
        config: cashocs.io.Config,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the mesh quality constraint for the triangle angles.

        Args:
            mesh: The corresponding FEM mesh.
            config: The configuration of the optimization problem.
            deformation_space: A space of vector CG1 elements for mesh deformations.

        """
        super().__init__(mesh, deformation_space)
        self.config = config

        self.dim = self.mesh.geometry().dim()
        self.min_angle = self.config.getfloat("MeshQualityConstraints", "min_angle")
        self.cells = self.mesh.cells()

        self.no_constraints = 3 * len(self.cells)
        self.is_necessary = np.array([True] * self.no_constraints)

    # ToDo: compute the minimum angle of each element directly and
    #  use this as threshold (scaled with a factor),
    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) <= 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`. The constraint returns :math:`\alpha_{min} - \alpha`,
        where :math:`\alpha` is one angle of a triangle and :math:`\alpha_{min}` is the
        minimum feasible angle in each triangle.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        old_coords = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coords_seq.reshape(-1, self.dim)
        self.mesh.bounding_box_tree().build(self.mesh)

        values = mesh_quality.triangle_angles(self.mesh)
        values = self.min_angle * 2 * np.pi / 360.0 - values

        self.mesh.coordinates()[:, :] = old_coords
        self.mesh.bounding_box_tree().build(self.mesh)

        return values

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        old_coords = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coords_seq.reshape(-1, self.dim)
        self.mesh.bounding_box_tree().build(self.mesh)

        rows, cols, vals = mesh_quality.triangle_angle_gradient(self.mesh)

        self.mesh.coordinates()[:, :] = old_coords
        self.mesh.bounding_box_tree().build(self.mesh)

        cols_local = self.v2d[cols]
        cols = self.l2g_dofs[cols_local]

        csr = rows, cols, vals
        shape = (int(3 * len(self.cells)), self.no_vertices * self.dim)
        gradient = sparse2scipy(csr, shape)

        return gradient


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
            self.function_values = np.concatenate(result)
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
        function_values = self.evaluate(coords_seq)
        return function_values[active_idx]

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
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
    ) -> sparse.csr_matrix:
        """Computes the gradient of those constraint functions that are active.

        Args:
            active_idx: The boolean array corresponding to the current working set.
            constraint_gradient: The entire (sparse) constraint gradient.

        Returns:
            A sparse matrix of the constraint gradient w.r.t. the working set.

        """
        scipy_matrix = constraint_gradient[active_idx]
        petsc_matrix = scipy2petsc(
            scipy_matrix, scipy_matrix.shape, self.mesh.mpi_comm()
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
        result = np.abs(function_values) <= self.constraint_tolerance
        return result

    def is_necessary(self, active_idx: np.ndarray) -> bool:
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
        return np.any(np.logical_and(self.necessary_constraints, active_idx))

    def is_feasible(self, coords_seq: np.ndarray) -> np.ndarray:
        """Checks, whether a given point is feasible w.r.t. all constraints.

        Args:
            coords_seq: The list of flattened mesh coordinates for the trial point.

        Returns:
            A boolean array, which is `True` in component `i` if the corresponding
            constraint is not violated, else `False`.

        """
        function_values = self.evaluate(coords_seq)
        result = np.less_equal(function_values, self.constraint_tolerance)
        return result


def sparse2scipy(
    csr: tuple[np.ndarray, np.ndarray, np.ndarray], shape=None
) -> sparse.csr_matrix:
    """Converts a sparse matrix representation to a sparse scipy matrix.

    Args:
        csr: The tuple making up the CSR matrix: `rows, cols, vals`.
        shape: The shape of the sparse matrix.

    Returns:
        The corresponding sparse scipy csr matrix.

    """
    rows = csr[0]
    cols = csr[1]
    vals = csr[2]
    A = sparse.csr_matrix((vals, (rows, cols)), shape=shape)
    return A


def scipy2petsc(scipy_matrix, shape, comm):
    no_rows_total = comm.allreduce(shape[0], op=MPI.SUM)
    petsc_matrix = PETSc.Mat().createAIJ(
        comm=comm,
        size=((shape[0], no_rows_total), shape[1]),
        csr=(scipy_matrix.indptr, scipy_matrix.indices, scipy_matrix.data),
    )

    return petsc_matrix


def sparse2petsc(
    csr: tuple[np.ndarray, np.ndarray, np.ndarray], shape=None, comm=None
) -> None:
    """Converts a sparse matrix representation to a sparse PETSc matrix.

    Args:
        csr: The tuple making up the CSR matrix: `rows, cols, vals`.

    """
    scipy_matrix = sparse2scipy(csr, shape)

    no_rows_total = comm.allreduce(shape[0], op=MPI.SUM)
    petsc_matrix = PETSc.Mat().createAIJ(
        comm=comm,
        size=((shape[0], no_rows_total), shape[1]),
        csr=(scipy_matrix.indptr, scipy_matrix.indices, scipy_matrix.data),
    )

    return petsc_matrix
