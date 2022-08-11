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

r"""Regularization for shape optimization problems.

This includes a (target) volume, surface, and barycenter regularization,
which are the :math:`L^2` distances between current volume, surface,
and barycenter, and desired ones.
"""

from __future__ import annotations

import json
from typing import Dict, List, TYPE_CHECKING

import fenics
import ufl

from cashocs import _loggers
from cashocs import _utils

if TYPE_CHECKING:
    from cashocs._forms import shape_form_handler


def t_grad(u: fenics.Function, n: fenics.FacetNormal) -> ufl.core.expr.Expr:
    """Computes the tangential gradient of u.

    Args:
        u: The argument, whose tangential gradient is to be computed.
        n: The unit outer normal vector.

    Returns:
        The tangential gradient of u.

    """
    return fenics.grad(u) - fenics.outer(fenics.grad(u) * n, n)


def t_div(u: fenics.Function, n: fenics.FacetNormal) -> ufl.core.expr.Expr:
    """Computes the tangential divergence of u.

    Args:
        u: The argument, whose tangential divergence is to be computed.
        n: The unit outer normal vector.

    Returns:
        The tangential divergence of u.

    """
    return fenics.div(u) - fenics.inner(fenics.grad(u) * n, n)


class ShapeRegularization:
    """Regularization terms for shape optimization problems."""

    def __init__(self, form_handler: shape_form_handler.ShapeFormHandler) -> None:
        """Initializes self.

        Args:
            form_handler: The corresponding shape form handler object.

        """
        self.test_vector_field = form_handler.test_vector_field
        self.config = form_handler.config
        self.geometric_dimension = form_handler.mesh.geometric_dimension()
        self.mesh = form_handler.mesh
        self.has_cashocs_remesh_flag = form_handler.has_cashocs_remesh_flag
        self.temp_dir = form_handler.temp_dir

        self.dx = fenics.Measure("dx", self.mesh)
        self.ds = fenics.Measure("ds", self.mesh)

        self.a_curvature_matrix = fenics.PETScMatrix()
        self.b_curvature = fenics.PETScVector()

        self.spatial_coordinate = fenics.SpatialCoordinate(self.mesh)

        self.use_relative_scaling = self.config.getboolean(
            "Regularization", "use_relative_scaling"
        )

        self.measure_hole = self.config.getboolean("Regularization", "measure_hole")
        if self.measure_hole:
            self.x_start = self.config.getfloat("Regularization", "x_start")
            self.x_end = self.config.getfloat("Regularization", "x_end")
            self.delta_x = self.x_end - self.x_start

            self.y_start = self.config.getfloat("Regularization", "y_start")
            self.y_end = self.config.getfloat("Regularization", "y_end")
            self.delta_y = self.y_end - self.y_start

            self.z_start = self.config.getfloat("Regularization", "z_start")
            self.z_end = self.config.getfloat("Regularization", "z_end")
            self.delta_z = self.z_end - self.z_start
            if self.geometric_dimension == 2:
                self.delta_z = 1.0

        self._init_volume_regularization()
        self._init_surface_regularization()
        self._init_curvature_regularization(form_handler)
        self._init_barycenter_regularization()

        if (
            self.mu_volume > 0.0
            or self.mu_surface > 0.0
            or self.mu_curvature > 0.0
            or self.mu_barycenter > 0.0
        ):
            self.has_regularization = True
        else:
            self.has_regularization = False

        self._scale_weights()

        self.current_volume = fenics.Expression("val", degree=0, val=1.0)
        self.current_surface = fenics.Expression("val", degree=0, val=1.0)
        self.current_barycenter_x = fenics.Expression("val", degree=0, val=0.0)
        self.current_barycenter_y = fenics.Expression("val", degree=0, val=0.0)
        self.current_barycenter_z = fenics.Expression("val", degree=0, val=0.0)

    def _init_volume_regularization(self) -> None:
        """Initializes the terms corresponding to the volume regularization."""
        self.mu_volume: float = self.config.getfloat("Regularization", "factor_volume")
        self.target_volume = self.config.getfloat("Regularization", "target_volume")
        if self.config.getboolean("Regularization", "use_initial_volume"):
            self.target_volume = self._compute_volume()

    def _init_surface_regularization(self) -> None:
        """Initializes the terms corresponding to the surface regularization."""
        self.mu_surface: float = self.config.getfloat(
            "Regularization", "factor_surface"
        )
        self.target_surface = self.config.getfloat("Regularization", "target_surface")
        if self.config.getboolean("Regularization", "use_initial_surface"):
            self.target_surface = fenics.assemble(fenics.Constant(1) * self.ds)

    def _init_curvature_regularization(
        self, form_handler: shape_form_handler.ShapeFormHandler
    ) -> None:
        """Initializes the terms corresponding to the surface regularization.

        Args:
            form_handler: The form handler of the problem.

        """
        self.mu_curvature: float = self.config.getfloat(
            "Regularization", "factor_curvature"
        )
        self.kappa_curvature = fenics.Function(form_handler.deformation_space)
        if self.mu_curvature > 0.0:
            n = fenics.FacetNormal(self.mesh)
            x = fenics.SpatialCoordinate(self.mesh)
            self.a_curvature = (
                fenics.inner(
                    fenics.TrialFunction(form_handler.deformation_space),
                    fenics.TestFunction(form_handler.deformation_space),
                )
                * self.ds
            )
            self.l_curvature = (
                fenics.inner(
                    t_grad(x, n),
                    t_grad(fenics.TestFunction(form_handler.deformation_space), n),
                )
                * self.ds
            )

    def _init_barycenter_regularization(self) -> None:
        """Initializes the terms corresponding to the barycenter regularization."""
        self.mu_barycenter: float = self.config.getfloat(
            "Regularization", "factor_barycenter"
        )

        self.target_barycenter_list = self.config.getlist(
            "Regularization", "target_barycenter"
        )

        if self.geometric_dimension == 2 and len(self.target_barycenter_list) == 2:
            self.target_barycenter_list.append(0.0)

        if self.config.getboolean("Regularization", "use_initial_barycenter"):
            self.target_barycenter_list = self._compute_barycenter_list()

    def update_geometric_quantities(self) -> None:
        """Updates the geometric quantities.

        Updates the volume, surface area, and barycenters (after the
        mesh is updated).
        """
        volume = self._compute_volume()
        barycenter_list = self._compute_barycenter_list()

        surface = fenics.assemble(fenics.Constant(1) * self.ds)

        self.current_volume.val = volume
        self.current_surface.val = surface
        self.current_barycenter_x.val = barycenter_list[0]
        self.current_barycenter_y.val = barycenter_list[1]
        self.current_barycenter_z.val = barycenter_list[2]

        self.compute_curvature()

    def compute_curvature(self) -> None:
        """Computes the mean curvature vector of the geometry."""
        if self.mu_curvature > 0.0:
            fenics.assemble(
                self.a_curvature, keep_diagonal=True, tensor=self.a_curvature_matrix
            )
            self.a_curvature_matrix.ident_zeros()

            fenics.assemble(self.l_curvature, tensor=self.b_curvature)

            _utils.solve_linear_problem(
                A=self.a_curvature_matrix.mat(),
                b=self.b_curvature.vec(),
                x=self.kappa_curvature.vector().vec(),
            )
            self.kappa_curvature.vector().apply("")

        else:
            pass

    def compute_objective(self) -> float:
        """Computes the part of the objective value that comes from the regularization.

        Returns:
            Part of the objective value coming from the regularization

        """
        if self.has_regularization:

            value = 0.0

            if self.mu_volume > 0.0:
                volume = self._compute_volume()
                value += 0.5 * self.mu_volume * pow(volume - self.target_volume, 2)

            if self.mu_surface > 0.0:
                surface = fenics.assemble(fenics.Constant(1.0) * self.ds)
                value += 0.5 * self.mu_surface * pow(surface - self.target_surface, 2)

            if self.mu_curvature > 0.0:
                self.compute_curvature()
                curvature_val = fenics.assemble(
                    fenics.inner(self.kappa_curvature, self.kappa_curvature) * self.ds
                )
                value += 0.5 * self.mu_curvature * curvature_val

            if self.mu_barycenter > 0.0:
                barycenter_list = self._compute_barycenter_list()

                value += (
                    0.5
                    * self.mu_barycenter
                    * (
                        pow(barycenter_list[0] - self.target_barycenter_list[0], 2)
                        + pow(barycenter_list[1] - self.target_barycenter_list[1], 2)
                        + pow(barycenter_list[2] - self.target_barycenter_list[2], 2)
                    )
                )

            return value

        else:
            return 0.0

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the part of the shape derivative that comes from the regularization.

        Returns:
            The weak form of the shape derivative coming from the regularization

        """
        vector_field = self.test_vector_field
        if self.has_regularization:

            x = fenics.SpatialCoordinate(self.mesh)
            n = fenics.FacetNormal(self.mesh)
            identity = fenics.Identity(self.geometric_dimension)

            shape_form = (
                fenics.Constant(self.mu_surface)
                * (self.current_surface - fenics.Constant(self.target_surface))
                * t_div(vector_field, n)
                * self.ds
            )

            shape_form += fenics.Constant(self.mu_curvature) * (
                fenics.inner(
                    (identity - (t_grad(x, n) + (t_grad(x, n)).T))
                    * t_grad(vector_field, n),
                    t_grad(self.kappa_curvature, n),
                )
                * self.ds
                + fenics.Constant(0.5)
                * t_div(vector_field, n)
                * t_div(self.kappa_curvature, n)
                * self.ds
            )

            if not self.measure_hole:
                shape_form += (
                    fenics.Constant(self.mu_volume)
                    * (self.current_volume - fenics.Constant(self.target_volume))
                    * fenics.div(vector_field)
                    * self.dx
                )
                shape_form += (
                    fenics.Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_x
                        - fenics.Constant(self.target_barycenter_list[0])
                    )
                    * (
                        self.current_barycenter_x
                        / self.current_volume
                        * fenics.div(vector_field)
                        + 1
                        / self.current_volume
                        * (
                            vector_field[0]
                            + self.spatial_coordinate[0] * fenics.div(vector_field)
                        )
                    )
                    * self.dx
                    + fenics.Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_y
                        - fenics.Constant(self.target_barycenter_list[1])
                    )
                    * (
                        self.current_barycenter_y
                        / self.current_volume
                        * fenics.div(vector_field)
                        + 1
                        / self.current_volume
                        * (
                            vector_field[1]
                            + self.spatial_coordinate[1] * fenics.div(vector_field)
                        )
                    )
                    * self.dx
                )

                if self.geometric_dimension == 3:
                    shape_form += (
                        fenics.Constant(self.mu_barycenter)
                        * (
                            self.current_barycenter_z
                            - fenics.Constant(self.target_barycenter_list[2])
                        )
                        * (
                            self.current_barycenter_z
                            / self.current_volume
                            * fenics.div(vector_field)
                            + 1
                            / self.current_volume
                            * (
                                vector_field[2]
                                + self.spatial_coordinate[2] * fenics.div(vector_field)
                            )
                        )
                        * self.dx
                    )

            else:
                shape_form -= (
                    fenics.Constant(self.mu_volume)
                    * (self.current_volume - fenics.Constant(self.target_volume))
                    * fenics.div(vector_field)
                    * self.dx
                )
                shape_form += (
                    fenics.Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_x
                        - fenics.Constant(self.target_barycenter_list[0])
                    )
                    * (
                        self.current_barycenter_x
                        / self.current_volume
                        * fenics.div(vector_field)
                        - 1
                        / self.current_volume
                        * (
                            vector_field[0]
                            + self.spatial_coordinate[0] * fenics.div(vector_field)
                        )
                    )
                    * self.dx
                    + fenics.Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_y
                        - fenics.Constant(self.target_barycenter_list[1])
                    )
                    * (
                        self.current_barycenter_y
                        / self.current_volume
                        * fenics.div(vector_field)
                        - 1
                        / self.current_volume
                        * (
                            vector_field[1]
                            + self.spatial_coordinate[1] * fenics.div(vector_field)
                        )
                    )
                    * self.dx
                )

                if self.geometric_dimension == 3:
                    shape_form += (
                        fenics.Constant(self.mu_barycenter)
                        * (
                            self.current_barycenter_z
                            - fenics.Constant(self.target_barycenter_list[2])
                        )
                        * (
                            self.current_barycenter_z
                            / self.current_volume
                            * fenics.div(vector_field)
                            - 1
                            / self.current_volume
                            * (
                                vector_field[2]
                                + self.spatial_coordinate[2] * fenics.div(vector_field)
                            )
                        )
                        * self.dx
                    )

            return shape_form

        else:
            dim = self.geometric_dimension
            return fenics.inner(fenics.Constant([0] * dim), vector_field) * self.dx

    def _scale_volume_term(self) -> None:
        """Scales the volume regularization parameter."""
        if self.mu_volume > 0.0:
            volume = self._compute_volume()
            value = 0.5 * pow(volume - self.target_volume, 2)

            if abs(value) < 1e-15:
                _loggers.info(
                    "The volume regularization vanishes for the initial "
                    "iteration. Multiplying this term with the factor you "
                    "supplied as weight."
                )
            else:
                self.mu_volume /= abs(value)

    def _scale_surface_term(self) -> None:
        """Scales the surface regularization term."""
        if self.mu_surface > 0.0:
            surface = fenics.assemble(fenics.Constant(1.0) * self.ds)
            value = 0.5 * pow(surface - self.target_surface, 2)

            if abs(value) < 1e-15:
                _loggers.info(
                    "The surface regularization vanishes for the initial "
                    "iteration. Multiplying this term with the factor you "
                    "supplied as weight."
                )
            else:
                self.mu_surface /= abs(value)

    def _scale_curvature_term(self) -> None:
        """Scales the curvature regularization term."""
        if self.mu_curvature > 0.0:
            self.compute_curvature()
            value = 0.5 * fenics.assemble(
                fenics.inner(self.kappa_curvature, self.kappa_curvature) * self.ds
            )

            if abs(value) < 1e-15:
                _loggers.info(
                    "The curvature regularization vanishes for the initial "
                    "iteration. Multiplying this term with the factor you "
                    "supplied as weight."
                )
            else:
                self.mu_curvature /= abs(value)

    def _scale_barycenter_term(self) -> None:
        """Scales the barycenter regularization term."""
        if self.mu_barycenter > 0.0:
            barycenter_list = self._compute_barycenter_list()

            value = 0.5 * (
                pow(barycenter_list[0] - self.target_barycenter_list[0], 2)
                + pow(barycenter_list[1] - self.target_barycenter_list[1], 2)
                + pow(barycenter_list[2] - self.target_barycenter_list[2], 2)
            )

            if abs(value) < 1e-15:
                _loggers.info(
                    "The barycenter regularization vanishes for the initial "
                    "iteration. Multiplying this term with the factor you "
                    "supplied as weight."
                )
            else:
                self.mu_barycenter /= abs(value)

    def _scale_weights(self) -> None:
        """Scales the terms of the regularization by the weights given in the config."""
        if self.use_relative_scaling and self.has_regularization:
            if not self.has_cashocs_remesh_flag:
                self._scale_volume_term()
                self._scale_surface_term()
                self._scale_curvature_term()
                self._scale_barycenter_term()

            else:

                with open(
                    f"{self.temp_dir}/temp_dict.json", "r", encoding="utf-8"
                ) as file:
                    temp_dict: Dict = json.load(file)

                self.mu_volume = temp_dict["Regularization"]["mu_volume"]
                self.mu_surface = temp_dict["Regularization"]["mu_surface"]
                self.mu_curvature = temp_dict["Regularization"]["mu_curvature"]
                self.mu_barycenter = temp_dict["Regularization"]["mu_barycenter"]

    def _compute_barycenter_list(self) -> List[float]:
        """Computes the list of barycenters for the geometry.

        Returns:
            The list of coordinates of the barycenter of the geometry.

        """
        barycenter_list = [0.0] * 3
        volume = self._compute_volume()

        if not self.measure_hole:
            barycenter_list[0] = (
                fenics.assemble(self.spatial_coordinate[0] * self.dx) / volume
            )
            barycenter_list[1] = (
                fenics.assemble(self.spatial_coordinate[1] * self.dx) / volume
            )
            if self.geometric_dimension == 3:
                barycenter_list[2] = (
                    fenics.assemble(self.spatial_coordinate[2] * self.dx) / volume
                )
            else:
                barycenter_list[2] = 0.0

            return barycenter_list

        else:
            barycenter_list[0] = (
                0.5
                * (pow(self.x_end, 2) - pow(self.x_start, 2))
                * self.delta_y
                * self.delta_z
                - fenics.assemble(self.spatial_coordinate[0] * self.dx)
            ) / volume
            barycenter_list[1] = (
                0.5
                * (pow(self.y_end, 2) - pow(self.y_start, 2))
                * self.delta_x
                * self.delta_z
                - fenics.assemble(self.spatial_coordinate[1] * self.dx)
            ) / volume
            if self.geometric_dimension == 3:
                barycenter_list[2] = (
                    0.5
                    * (pow(self.z_end, 2) - pow(self.z_start, 2))
                    * self.delta_x
                    * self.delta_y
                    - fenics.assemble(self.spatial_coordinate[2] * self.dx)
                ) / volume
            else:
                barycenter_list[2] = 0.0

            return barycenter_list

    def _compute_volume(self) -> float:
        """Computes the volume of the geometry.

        Returns:
            The volume of the geometry.

        """
        volume: float
        if not self.measure_hole:
            volume = fenics.assemble(fenics.Constant(1.0) * self.dx)
        else:
            volume = self.delta_x * self.delta_y * self.delta_z - fenics.assemble(
                fenics.Constant(1) * self.dx
            )

        return volume
