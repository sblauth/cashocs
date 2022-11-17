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

"""Management of mesh deformations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import fenics
import numpy as np

from cashocs import _exceptions
from cashocs import _loggers
from cashocs.geometry import measure

if TYPE_CHECKING:
    from cashocs.geometry import mesh_testing


class DeformationHandler:
    """A class, which implements mesh deformations.

    The deformations can be due to a deformation vector field or a (piecewise) update of
    the mesh coordinates.
    """

    def __init__(
        self,
        mesh: fenics.Mesh,
        a_priori_tester: mesh_testing.APrioriMeshTester,
        a_posteriori_tester: mesh_testing.APosterioriMeshTester,
    ) -> None:
        """Initializes self.

        Args:
            mesh: The fenics mesh which is to be deformed.
            a_priori_tester: The tester before mesh modification.
            a_posteriori_tester: The tester after mesh modification.

        """
        self.mesh = mesh
        self.a_priori_tester = a_priori_tester
        self.a_posteriori_tester = a_posteriori_tester

        self.dim = self.mesh.geometric_dimension()
        self.dx = measure.NamedMeasure("dx", self.mesh)
        self.old_coordinates = self.mesh.coordinates().copy()
        self.shape_coordinates = self.old_coordinates.shape
        self.vector_cg_space = fenics.VectorFunctionSpace(mesh, "CG", 1)
        self.dg_function_space = fenics.FunctionSpace(mesh, "DG", 0)
        self.bbtree = self.mesh.bounding_box_tree()

        self.v2d = fenics.vertex_to_dof_map(self.vector_cg_space).reshape(
            (-1, self.mesh.geometry().dim())
        )
        self.d2v = fenics.dof_to_vertex_map(self.vector_cg_space)

        self.coordinates = self.mesh.coordinates()

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
            if not self.a_priori_tester.test(dof_transformation, float("inf")):
                _loggers.debug(
                    "Mesh transformation rejected due to a priori check.\n"
                    "Reason: Transformation would result in inverted mesh elements."
                )
                return False
            else:
                self.old_coordinates = self.mesh.coordinates().copy()
                self.coordinates += coordinate_transformation
                self.bbtree.build(self.mesh)

                check = self.a_posteriori_tester.test()
                if not check:
                    self.revert_transformation()

                return check
        else:
            self.old_coordinates = self.mesh.coordinates().copy()
            self.coordinates += coordinate_transformation
            self.bbtree.build(self.mesh)

            check = self.a_posteriori_tester.test()
            if not check:
                self.revert_transformation()

            return check

    def coordinate_to_dof(self, coordinate_deformation: np.ndarray) -> fenics.Function:
        """Converts a coordinate deformation to a deformation vector field (dof based).

        Args:
            coordinate_deformation: The deformation for the mesh coordinates.

        Returns:
            The deformation vector field.

        """
        dof_vector = coordinate_deformation.reshape(-1)[self.d2v]
        dof_deformation = fenics.Function(self.vector_cg_space)
        dof_deformation.vector().set_local(dof_vector)
        dof_deformation.vector().apply("")

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

        vertex_values = dof_deformation.compute_vertex_values()
        coordinate_deformation: np.ndarray = vertex_values.reshape(self.dim, -1).T
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

        check = self.a_posteriori_tester.test()
        if not check:
            self.revert_transformation()

        return check
