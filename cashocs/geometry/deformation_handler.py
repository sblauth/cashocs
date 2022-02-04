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

"""Module for managing mesh deformations."""

from __future__ import annotations

import collections
from typing import Union

import fenics
import numpy as np
from petsc4py import PETSc

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import utils
from cashocs.geometry import measure


class DeformationHandler:
    """A class, which implements mesh deformations.

    The deformations can be due to a deformation vector field or a (piecewise) update of
    the mesh coordinates.
    """

    def __init__(self, mesh: fenics.Mesh) -> None:
        """
        Args:
            mesh: The fenics mesh which is to be deformed.
        """

        self.mesh = mesh
        self.dx = measure._NamedMeasure("dx", self.mesh)
        self.old_coordinates = self.mesh.coordinates().copy()
        self.shape_coordinates = self.old_coordinates.shape
        self.VCG = fenics.VectorFunctionSpace(mesh, "CG", 1)
        self.DG0 = fenics.FunctionSpace(mesh, "DG", 0)
        self.bbtree = self.mesh.bounding_box_tree()
        self._setup_a_priori()
        self.v2d = fenics.vertex_to_dof_map(self.VCG).reshape(
            (-1, self.mesh.geometry().dim())
        )
        self.d2v = fenics.dof_to_vertex_map(self.VCG)

        cells = self.mesh.cells()
        flat_cells = cells.flatten().tolist()
        self.cell_counter = collections.Counter(flat_cells)
        self.occurrences = np.array(
            [self.cell_counter[i] for i in range(self.mesh.num_vertices())]
        )
        self.coordinates = self.mesh.coordinates()

    def _setup_a_priori(self) -> None:
        """Sets up the attributes and petsc solver for the a priori quality check."""

        self.options_prior = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]
        # noinspection PyUnresolvedReferences
        self.ksp_prior = PETSc.KSP().create()
        utils._setup_petsc_options([self.ksp_prior], [self.options_prior])

        self.transformation_container = fenics.Function(self.VCG)
        dim = self.mesh.geometric_dimension()

        self.a_prior = (
            fenics.TrialFunction(self.DG0) * fenics.TestFunction(self.DG0) * self.dx
        )
        self.L_prior = (
            fenics.det(
                fenics.Identity(dim) + fenics.grad(self.transformation_container)
            )
            * fenics.TestFunction(self.DG0)
            * self.dx
        )

    def _test_a_priori(self, transformation: fenics.Function) -> bool:
        r"""Check the quality of the transformation before the actual mesh is moved.

        Checks the quality of the transformation. The criterion is that

        .. math:: \det(I + D \texttt{transformation})

        should neither be too large nor too small in order to achieve the best
        transformations.

        Args:
            transformation: The transformation for the mesh.

        Returns:
            A boolean that indicates whether the desired transformation is feasible.
        """

        self.transformation_container.vector().vec().aypx(
            0.0, transformation.vector().vec()
        )
        # noinspection PyPep8Naming
        x = utils._assemble_and_solve_linear(
            self.a_prior,
            self.L_prior,
            ksp=self.ksp_prior,
            ksp_options=self.options_prior,
        )
        min_det = np.min(x[:])

        return min_det > 0

    def _test_a_posteriori(self) -> bool:
        """Checks the quality of the transformation after the actual mesh is moved.

        Checks whether the mesh is a valid finite element mesh
        after it has been moved, i.e., if there are no overlapping
        or self intersecting elements.

        Returns:
            True if the test is successful, False otherwise.

        Notes:
            fenics itself does not check whether the used mesh is a valid finite
            element mesh, so this check has to be done manually.
        """

        self_intersections = False
        collisions = CollisionCounter.compute_collisions(self.mesh)
        if not (collisions == self.occurrences).all():
            self_intersections = True

        if self_intersections:
            self.revert_transformation()
            _loggers.debug("Mesh transformation rejected due to a posteriori check.")
            return False
        else:
            return True

    def revert_transformation(self) -> None:
        """Reverts the previous mesh transformation.

        This is used when the mesh quality for the resulting deformed mesh
        is not sufficient, or when the solution algorithm terminates, e.g., due
        to lack of sufficient decrease in the Armijo rule
        """

        self.mesh.coordinates()[:, :] = self.old_coordinates
        del self.old_coordinates
        self.bbtree.build(self.mesh)

    def move_mesh(
        self,
        transformation: Union[fenics.Function, np.ndarray],
        validated_a_priori: bool = False,
    ) -> bool:
        r"""Transforms the mesh by perturbation of identity.

        Moves the mesh according to the deformation given by

        .. math:: \text{id} + \mathcal{V}(x),

        where :math:`\mathcal{V}` is the transformation. This
        represents the perturbation of identity.

        Args:
            transformation: The transformation for the mesh, a vector CG1 Function.
            validated_a_priori: A boolean flag, which indicates whether an a-priori
                check has already been performed before moving the mesh. Default is
                ``False``

        Returns:
            ``True`` if the mesh movement was successful, ``False`` otherwise.
        """

        if isinstance(transformation, np.ndarray):
            if not transformation.shape == self.coordinates.shape:
                raise _exceptions.CashocsException(
                    "Not a valid dimension for the transformation"
                )
            else:
                coordinate_transformation = transformation
        else:
            coordinate_transformation = self.dof_to_coordinate(transformation)

        if not validated_a_priori:
            if isinstance(transformation, np.ndarray):
                dof_transformation = self.coordinate_to_dof(transformation)
            else:
                dof_transformation = transformation
            if not self._test_a_priori(dof_transformation):
                _loggers.debug(
                    "Mesh transformation rejected due to a priori check.\n"
                    "Reason: Transformation would result in inverted mesh elements."
                )
                return False
            else:
                self.old_coordinates = self.mesh.coordinates().copy()
                self.coordinates += coordinate_transformation
                self.bbtree.build(self.mesh)

                return self._test_a_posteriori()
        else:
            self.old_coordinates = self.mesh.coordinates().copy()
            self.coordinates += coordinate_transformation
            self.bbtree.build(self.mesh)

            return self._test_a_posteriori()

    def coordinate_to_dof(self, coordinate_deformation: np.ndarray) -> fenics.Function:
        """Converts a coordinate deformation to a deformation vector field (dof based).

        Args:
            coordinate_deformation: The deformation for the mesh coordinates.

        Returns:
            The deformation vector field.
        """

        dof_vector = coordinate_deformation.reshape(-1)[self.d2v]
        dof_deformation = fenics.Function(self.VCG)
        dof_deformation.vector()[:] = dof_vector

        return dof_deformation

    def dof_to_coordinate(self, dof_deformation: fenics.Function) -> np.ndarray:
        """Converts a deformation vector field to a coordinate based deformation.

        Args:
            dof_deformation: The deformation vector field.

        Returns:
            The array which can be used to deform the mesh coordinates.
        """

        if not (
            dof_deformation.ufl_element().family() == "Lagrange"
            and dof_deformation.ufl_element().degree() == 1
        ):
            raise _exceptions.InputError(
                "cashocs.geometry.DeformationHandler.dof_to_coordinate",
                "dof_deformation",
                "dof_deformation has to be a piecewise linear Lagrange vector field.",
            )

        coordinate_deformation = dof_deformation.vector().vec()[self.v2d]

        return coordinate_deformation

    def assign_coordinates(self, coordinates: np.ndarray) -> bool:
        """Assigns coordinates to self.mesh.

        Args:
            coordinates: Array of mesh coordinates, which you want to assign.

        Returns:
            ``True`` if the assignment was possible, ``False`` if not.
        """

        self.old_coordinates = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coordinates[:, :]
        self.bbtree.build(self.mesh)

        return self._test_a_posteriori()


class CollisionCounter:
    """Class for testing, whether a given mesh is a valid FEM mesh."""

    _cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/Point.h>

using namespace dolfin;

Eigen::VectorXi
compute_collisions(std::shared_ptr<const Mesh> mesh)
{
  int num_vertices;
  std::vector<unsigned int> colliding_cells;

  num_vertices = mesh->num_vertices();
  Eigen::VectorXi collisions(num_vertices);

  int i = 0;
  for (VertexIterator v(*mesh); !v.end(); ++v)
  {
    colliding_cells = mesh->bounding_box_tree()->compute_entity_collisions(
      v->point()
    );
    collisions[i] = colliding_cells.size();

    ++i;
  }
  return collisions;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("compute_collisions", &compute_collisions);
}
"""
    _cpp_object = fenics.compile_cpp_code(_cpp_code)

    def __init__(self) -> None:
        pass

    @classmethod
    def compute_collisions(cls, mesh: fenics.Mesh) -> np.ndarray:
        """Computes the cells which (potentially) contain self intersections.

        Args:
            mesh: A FEM mesh.

        Returns:
            An array of cell indices, where array[i] contains the indices of all cells
            that vertex i collides with.
        """

        return cls._cpp_object.compute_collisions(mesh)
