# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

r"""Regularization for shape optimization problems.

This includes a (target) volume, surface, and barycenter regularization,
which are the :math:`L^2` distances between current volume, surface,
and barycenter, and desired ones.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import fenics
import ufl
from fenics import Constant, div, inner

from .._loggers import info
from ..utils import _solve_linear_problem


if TYPE_CHECKING:
    from .._forms import ShapeFormHandler


def t_grad(u: fenics.Function, n: fenics.FacetNormal) -> ufl.core.expr.Expr:
    """Computes the tangential gradient of u

    Parameters
    ----------
    u : fenics.Function
        the argument
    n : fenics.FacetNormal
        the unit outer normal vector

    Returns
    -------
    ufl.core.expr.Expr
        the tangential gradient of u
    """

    return fenics.grad(u) - fenics.outer(fenics.grad(u) * n, n)


def t_div(u: fenics.Function, n: fenics.FacetNormal) -> ufl.core.expr.Expr:
    """Computes the tangential divergence of u

    Parameters
    ----------
    u : fenics.Function
        the argument
    n : fenics.FacetNormal
        the outer unit normal vector

    Returns
    -------
    ufl.core.expr.Expr
        the tangential divergence of u
    """

    return fenics.div(u) - fenics.inner(fenics.grad(u) * n, n)


class Regularization:
    """Regularization terms for shape optimization problems"""

    def __init__(self, form_handler: ShapeFormHandler) -> None:
        """Initializes the regularization

        Parameters
        ----------
        form_handler : ShapeFormHandler
            the corresponding shape form handler object
        """

        self.test_vector_field = form_handler.test_vector_field
        self.config = form_handler.config
        self.geometric_dimension = form_handler.mesh.geometric_dimension()
        self.mesh = form_handler.mesh
        self.has_cashocs_remesh_flag = form_handler.has_cashocs_remesh_flag
        self.temp_dir = form_handler.temp_dir

        self.dx = fenics.Measure("dx", self.mesh)
        self.ds = fenics.Measure("ds", self.mesh)

        self.A_curvature = fenics.PETScMatrix()
        self.b_curvature = fenics.PETScVector()

        self.spatial_coordinate = fenics.SpatialCoordinate(self.mesh)

        self.use_relative_scaling = self.config.getboolean(
            "Regularization", "use_relative_scaling", fallback=False
        )

        self.measure_hole = self.config.getboolean(
            "Regularization", "measure_hole", fallback=False
        )
        if self.measure_hole:
            self.x_start = self.config.getfloat(
                "Regularization", "x_start", fallback=0.0
            )
            self.x_end = self.config.getfloat("Regularization", "x_end", fallback=1.0)
            if not self.x_end >= self.x_start:
                raise IncompatibleConfigurationError(
                    "x_start",
                    "Regularization",
                    "x_end",
                    "Regularization",
                    "Reason: x_end must not be smaller than x_start.",
                )
            self.delta_x = self.x_end - self.x_start

            self.y_start = self.config.getfloat(
                "Regularization", "y_start", fallback=0.0
            )
            self.y_end = self.config.getfloat("Regularization", "y_end", fallback=1.0)
            if not self.y_end >= self.y_start:
                raise IncompatibleConfigurationError(
                    "y_start",
                    "Regularization",
                    "y_end",
                    "Regularization",
                    "Reason: y_end must not be smaller than y_start.",
                )
            self.delta_y = self.y_end - self.y_start

            self.z_start = self.config.getfloat(
                "Regularization", "z_start", fallback=0.0
            )
            self.z_end = self.config.getfloat("Regularization", "z_end", fallback=1.0)
            if not self.z_end >= self.z_start:
                raise IncompatibleConfigurationError(
                    "z_start",
                    "Regularization",
                    "z_end",
                    "Regularization",
                    "Reason: z_end must not be smaller than z_start.",
                )
            self.delta_z = self.z_end - self.z_start
            if self.geometric_dimension == 2:
                self.delta_z = 1.0

        self.mu_volume = self.config.getfloat(
            "Regularization", "factor_volume", fallback=0.0
        )
        self.target_volume = self.config.getfloat(
            "Regularization", "target_volume", fallback=0.0
        )
        if self.config.getboolean(
            "Regularization", "use_initial_volume", fallback=False
        ):
            if not self.measure_hole:
                self.target_volume = fenics.assemble(Constant(1) * self.dx)
            else:
                self.target_volume = (
                    self.delta_x * self.delta_y * self.delta_z
                    - fenics.assemble(Constant(1.0) * self.dx)
                )

        self.mu_surface = self.config.getfloat(
            "Regularization", "factor_surface", fallback=0.0
        )
        self.target_surface = self.config.getfloat(
            "Regularization", "target_surface", fallback=0.0
        )
        if self.config.getboolean(
            "Regularization", "use_initial_surface", fallback=False
        ):
            self.target_surface = fenics.assemble(Constant(1) * self.ds)

        self.mu_curvature = self.config.getfloat(
            "Regularization", "factor_curvature", fallback=0.0
        )
        self.kappa_curvature = fenics.Function(form_handler.deformation_space)
        if self.mu_curvature > 0.0:
            n = fenics.FacetNormal(self.mesh)
            x = fenics.SpatialCoordinate(self.mesh)
            self.a_curvature = (
                inner(
                    fenics.TrialFunction(form_handler.deformation_space),
                    fenics.TestFunction(form_handler.deformation_space),
                )
                * self.ds
            )
            self.L_curvature = (
                inner(
                    t_grad(x, n),
                    t_grad(fenics.TestFunction(form_handler.deformation_space), n),
                )
                * self.ds
            )

        self.mu_barycenter = self.config.getfloat(
            "Regularization", "factor_barycenter", fallback=0.0
        )

        target_bar = self.config.get(
            "Regularization", "target_barycenter", fallback="[0,0,0]"
        )
        self.target_barycenter_list = json.loads(target_bar)

        if self.geometric_dimension == 2 and len(self.target_barycenter_list) == 2:
            self.target_barycenter_list.append(0.0)

        if self.config.getboolean(
            "Regularization", "use_initial_barycenter", fallback=False
        ):
            self.target_barycenter_list = [0.0, 0.0, 0.0]
            if not self.measure_hole:
                volume = fenics.assemble(Constant(1) * self.dx)
                self.target_barycenter_list[0] = (
                    fenics.assemble(self.spatial_coordinate[0] * self.dx) / volume
                )
                self.target_barycenter_list[1] = (
                    fenics.assemble(self.spatial_coordinate[1] * self.dx) / volume
                )
                if self.geometric_dimension == 3:
                    self.target_barycenter_list[2] = (
                        fenics.assemble(self.spatial_coordinate[2] * self.dx) / volume
                    )
                else:
                    self.target_barycenter_list[2] = 0.0

            else:
                volume = self.delta_x * self.delta_y * self.delta_z - fenics.assemble(
                    Constant(1) * self.dx
                )
                self.target_barycenter_list[0] = (
                    0.5
                    * (pow(self.x_end, 2) - pow(self.x_start, 2))
                    * self.delta_y
                    * self.delta_z
                    - fenics.assemble(self.spatial_coordinate[0] * self.dx)
                ) / volume
                self.target_barycenter_list[1] = (
                    0.5
                    * (pow(self.y_end, 2) - pow(self.y_start, 2))
                    * self.delta_x
                    * self.delta_z
                    - fenics.assemble(self.spatial_coordinate[1] * self.dx)
                ) / volume
                if self.geometric_dimension == 3:
                    self.target_barycenter_list[2] = (
                        0.5
                        * (pow(self.z_end, 2) - pow(self.z_start, 2))
                        * self.delta_x
                        * self.delta_y
                        - fenics.assemble(self.spatial_coordinate[2] * self.dx)
                    ) / volume
                else:
                    self.target_barycenter_list[2] = 0.0

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

    def update_geometric_quantities(self) -> None:
        """Updates the geometric quantities

        Updates the volume, surface area, and barycenters (after the
        mesh is updated)

        Returns
        -------
        None
        """

        if not self.measure_hole:
            volume = fenics.assemble(Constant(1) * self.dx)
            barycenter_x = (
                fenics.assemble(self.spatial_coordinate[0] * self.dx) / volume
            )
            barycenter_y = (
                fenics.assemble(self.spatial_coordinate[1] * self.dx) / volume
            )
            if self.geometric_dimension == 3:
                barycenter_z = (
                    fenics.assemble(self.spatial_coordinate[2] * self.dx) / volume
                )
            else:
                barycenter_z = 0.0

        else:
            volume = self.delta_x * self.delta_y * self.delta_z - fenics.assemble(
                Constant(1) * self.dx
            )
            barycenter_x = (
                0.5
                * (pow(self.x_end, 2) - pow(self.x_start, 2))
                * self.delta_y
                * self.delta_z
                - fenics.assemble(self.spatial_coordinate[0] * self.dx)
            ) / volume
            barycenter_y = (
                0.5
                * (pow(self.y_end, 2) - pow(self.y_start, 2))
                * self.delta_x
                * self.delta_z
                - fenics.assemble(self.spatial_coordinate[1] * self.dx)
            ) / volume
            if self.geometric_dimension == 3:
                barycenter_z = (
                    0.5
                    * (pow(self.z_end, 2) - pow(self.z_start, 2))
                    * self.delta_x
                    * self.delta_y
                    - fenics.assemble(self.spatial_coordinate[2] * self.dx)
                ) / volume
            else:
                barycenter_z = 0.0

        surface = fenics.assemble(Constant(1) * self.ds)

        self.current_volume.val = volume
        self.current_surface.val = surface
        self.current_barycenter_x.val = barycenter_x
        self.current_barycenter_y.val = barycenter_y
        self.current_barycenter_z.val = barycenter_z

        self.compute_curvature()

    def compute_curvature(self) -> None:
        """Computes the mean curvature vector of the geometry.

        Returns
        -------
        None
        """

        if self.mu_curvature > 0.0:
            fenics.assemble(
                self.a_curvature, keep_diagonal=True, tensor=self.A_curvature
            )
            self.A_curvature.ident_zeros()

            fenics.assemble(self.L_curvature, tensor=self.b_curvature)

            _solve_linear_problem(
                A=self.A_curvature.mat(),
                b=self.b_curvature.vec(),
                x=self.kappa_curvature.vector().vec(),
            )

        else:
            pass

    def compute_objective(self) -> float:
        """Computes the part of the objective value that comes from the regularization

        Returns
        -------
        float
            Part of the objective value coming from the regularization

        """

        if self.has_regularization:

            value = 0.0

            if self.mu_volume > 0.0:
                if not self.measure_hole:
                    volume = fenics.assemble(Constant(1.0) * self.dx)
                else:
                    volume = (
                        self.delta_x * self.delta_y * self.delta_z
                        - fenics.assemble(Constant(1) * self.dx)
                    )

                value += 0.5 * self.mu_volume * pow(volume - self.target_volume, 2)

            if self.mu_surface > 0.0:
                surface = fenics.assemble(Constant(1.0) * self.ds)
                # self.current_surface.val = surface
                value += 0.5 * self.mu_surface * pow(surface - self.target_surface, 2)

            if self.mu_curvature > 0.0:
                self.compute_curvature()
                curvature_val = fenics.assemble(
                    fenics.inner(self.kappa_curvature, self.kappa_curvature) * self.ds
                )
                value += 0.5 * self.mu_curvature * curvature_val

            if self.mu_barycenter > 0.0:
                if not self.measure_hole:
                    volume = fenics.assemble(Constant(1) * self.dx)

                    barycenter_x = (
                        fenics.assemble(self.spatial_coordinate[0] * self.dx) / volume
                    )
                    barycenter_y = (
                        fenics.assemble(self.spatial_coordinate[1] * self.dx) / volume
                    )
                    if self.geometric_dimension == 3:
                        barycenter_z = (
                            fenics.assemble(self.spatial_coordinate[2] * self.dx)
                            / volume
                        )
                    else:
                        barycenter_z = 0.0

                else:
                    volume = (
                        self.delta_x * self.delta_y * self.delta_z
                        - fenics.assemble(Constant(1) * self.dx)
                    )

                    barycenter_x = (
                        0.5
                        * (pow(self.x_end, 2) - pow(self.x_start, 2))
                        * self.delta_y
                        * self.delta_z
                        - fenics.assemble(self.spatial_coordinate[0] * self.dx)
                    ) / volume
                    barycenter_y = (
                        0.5
                        * (pow(self.y_end, 2) - pow(self.y_start, 2))
                        * self.delta_x
                        * self.delta_z
                        - fenics.assemble(self.spatial_coordinate[1] * self.dx)
                    ) / volume
                    if self.geometric_dimension == 3:
                        barycenter_z = (
                            0.5
                            * (pow(self.z_end, 2) - pow(self.z_start, 2))
                            * self.delta_x
                            * self.delta_y
                            - fenics.assemble(self.spatial_coordinate[2] * self.dx)
                        ) / volume
                    else:
                        barycenter_z = 0.0

                value += (
                    0.5
                    * self.mu_barycenter
                    * (
                        pow(barycenter_x - self.target_barycenter_list[0], 2)
                        + pow(barycenter_y - self.target_barycenter_list[1], 2)
                        + pow(barycenter_z - self.target_barycenter_list[2], 2)
                    )
                )

            return value

        else:
            return 0.0

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the part of the shape derivative that comes from the regularization

        Returns
        -------
        ufl.Form
            The weak form of the shape derivative coming from the regularization

        """

        V = self.test_vector_field
        if self.has_regularization:

            x = fenics.SpatialCoordinate(self.mesh)
            n = fenics.FacetNormal(self.mesh)
            I = fenics.Identity(self.geometric_dimension)

            self.shape_form = (
                Constant(self.mu_surface)
                * (self.current_surface - Constant(self.target_surface))
                * t_div(V, n)
                * self.ds
            )

            self.shape_form += Constant(self.mu_curvature) * (
                inner(
                    (I - (t_grad(x, n) + (t_grad(x, n)).T)) * t_grad(V, n),
                    t_grad(self.kappa_curvature, n),
                )
                * self.ds
                + Constant(0.5) * t_div(V, n) * t_div(self.kappa_curvature, n) * self.ds
            )

            if not self.measure_hole:
                self.shape_form += (
                    Constant(self.mu_volume)
                    * (self.current_volume - Constant(self.target_volume))
                    * div(V)
                    * self.dx
                )
                self.shape_form += (
                    Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_x
                        - Constant(self.target_barycenter_list[0])
                    )
                    * (
                        self.current_barycenter_x / self.current_volume * div(V)
                        + 1
                        / self.current_volume
                        * (V[0] + self.spatial_coordinate[0] * div(V))
                    )
                    * self.dx
                    + Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_y
                        - Constant(self.target_barycenter_list[1])
                    )
                    * (
                        self.current_barycenter_y / self.current_volume * div(V)
                        + 1
                        / self.current_volume
                        * (V[1] + self.spatial_coordinate[1] * div(V))
                    )
                    * self.dx
                )

                if self.geometric_dimension == 3:
                    self.shape_form += (
                        Constant(self.mu_barycenter)
                        * (
                            self.current_barycenter_z
                            - Constant(self.target_barycenter_list[2])
                        )
                        * (
                            self.current_barycenter_z / self.current_volume * div(V)
                            + 1
                            / self.current_volume
                            * (V[2] + self.spatial_coordinate[2] * div(V))
                        )
                        * self.dx
                    )

            else:
                self.shape_form -= (
                    Constant(self.mu_volume)
                    * (self.current_volume - Constant(self.target_volume))
                    * div(V)
                    * self.dx
                )
                self.shape_form += (
                    Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_x
                        - Constant(self.target_barycenter_list[0])
                    )
                    * (
                        self.current_barycenter_x / self.current_volume * div(V)
                        - 1
                        / self.current_volume
                        * (V[0] + self.spatial_coordinate[0] * div(V))
                    )
                    * self.dx
                    + Constant(self.mu_barycenter)
                    * (
                        self.current_barycenter_y
                        - Constant(self.target_barycenter_list[1])
                    )
                    * (
                        self.current_barycenter_y / self.current_volume * div(V)
                        - 1
                        / self.current_volume
                        * (V[1] + self.spatial_coordinate[1] * div(V))
                    )
                    * self.dx
                )

                if self.geometric_dimension == 3:
                    self.shape_form += (
                        Constant(self.mu_barycenter)
                        * (
                            self.current_barycenter_z
                            - Constant(self.target_barycenter_list[2])
                        )
                        * (
                            self.current_barycenter_z / self.current_volume * div(V)
                            - 1
                            / self.current_volume
                            * (V[2] + self.spatial_coordinate[2] * div(V))
                        )
                        * self.dx
                    )

            return self.shape_form

        else:
            dim = self.geometric_dimension
            return inner(fenics.Constant([0] * dim), V) * self.dx

    def _scale_weights(self):
        """Scales the terms of the regularization by the weights given in the config file

        Returns
        -------
        None
        """

        if self.use_relative_scaling and self.has_regularization:

            if not self.has_cashocs_remesh_flag:

                if self.mu_volume > 0.0:
                    if not self.measure_hole:
                        volume = fenics.assemble(Constant(1.0) * self.dx)
                    else:
                        volume = (
                            self.delta_x * self.delta_y * self.delta_z
                            - fenics.assemble(Constant(1) * self.dx)
                        )

                    value = 0.5 * pow(volume - self.target_volume, 2)

                    if abs(value) < 1e-15:
                        info(
                            "The volume regularization vanishes for the initial iteration. Multiplying this term with the factor you supplied as weight."
                        )
                    else:
                        self.mu_volume /= abs(value)

                if self.mu_surface > 0.0:
                    surface = fenics.assemble(Constant(1.0) * self.ds)
                    value = 0.5 * pow(surface - self.target_surface, 2)

                    if abs(value) < 1e-15:
                        info(
                            "The surface regularization vanishes for the initial iteration. Multiplying this term with the factor you supplied as weight."
                        )
                    else:
                        self.mu_surface /= abs(value)

                if self.mu_curvature > 0.0:
                    self.compute_curvature()
                    value = 0.5 * fenics.assemble(
                        fenics.inner(self.kappa_curvature, self.kappa_curvature)
                        * self.ds
                    )

                    if abs(value) < 1e-15:
                        info(
                            "The curvature regularization vanishes for the initial iteration. Multiplying this term with the factor you supplied as weight."
                        )
                    else:
                        self.mu_curvature /= abs(value)

                if self.mu_barycenter > 0.0:
                    if not self.measure_hole:
                        volume = fenics.assemble(Constant(1) * self.dx)

                        barycenter_x = (
                            fenics.assemble(self.spatial_coordinate[0] * self.dx)
                            / volume
                        )
                        barycenter_y = (
                            fenics.assemble(self.spatial_coordinate[1] * self.dx)
                            / volume
                        )
                        if self.geometric_dimension == 3:
                            barycenter_z = (
                                fenics.assemble(self.spatial_coordinate[2] * self.dx)
                                / volume
                            )
                        else:
                            barycenter_z = 0.0

                    else:
                        volume = (
                            self.delta_x * self.delta_y * self.delta_z
                            - fenics.assemble(Constant(1) * self.dx)
                        )

                        barycenter_x = (
                            0.5
                            * (pow(self.x_end, 2) - pow(self.x_start, 2))
                            * self.delta_y
                            * self.delta_z
                            - fenics.assemble(self.spatial_coordinate[0] * self.dx)
                        ) / volume
                        barycenter_y = (
                            0.5
                            * (pow(self.y_end, 2) - pow(self.y_start, 2))
                            * self.delta_x
                            * self.delta_z
                            - fenics.assemble(self.spatial_coordinate[1] * self.dx)
                        ) / volume
                        if self.geometric_dimension == 3:
                            barycenter_z = (
                                0.5
                                * (pow(self.z_end, 2) - pow(self.z_start, 2))
                                * self.delta_x
                                * self.delta_y
                                - fenics.assemble(self.spatial_coordinate[2] * self.dx)
                            ) / volume
                        else:
                            barycenter_z = 0.0

                    value = 0.5 * (
                        pow(barycenter_x - self.target_barycenter_list[0], 2)
                        + pow(barycenter_y - self.target_barycenter_list[1], 2)
                        + pow(barycenter_z - self.target_barycenter_list[2], 2)
                    )

                    if abs(value) < 1e-15:
                        info(
                            "The barycenter regularization vanishes for the initial iteration. Multiplying this term with the factor you supplied as weight."
                        )
                    else:
                        self.mu_barycenter /= abs(value)

            else:

                with open(f"{self.temp_dir}/temp_dict.json", "r") as file:
                    temp_dict = json.load(file)

                self.mu_volume = temp_dict["Regularization"]["mu_volume"]
                self.mu_surface = temp_dict["Regularization"]["mu_surface"]
                self.mu_curvature = temp_dict["Regularization"]["mu_curvature"]
                self.mu_barycenter = temp_dict["Regularization"]["mu_barycenter"]
