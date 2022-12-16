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

import abc
from typing import List, TYPE_CHECKING

import fenics
import ufl

from cashocs import _loggers
from cashocs import _utils

if TYPE_CHECKING:
    import ufl.core.expr

    from cashocs._database import database


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


class ShapeRegularizationTerm(abc.ABC):
    """Regularization terms for shape optimization."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        self.db = db

        self.config = self.db.config
        self.mesh = db.geometry_db.mesh
        self.dx = fenics.Measure("dx", self.mesh)
        self.use_relative_scaling = self.config.getboolean(
            "Regularization", "use_relative_scaling"
        )
        self.is_active = False
        self.test_vector_field = fenics.TestFunction(
            self.db.function_db.control_spaces[0]
        )

    @abc.abstractmethod
    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the shape derivative of the regularization term.

        Returns:
            The ufl form of the shape derivative.

        """
        pass

    def update(self) -> None:
        """Updates the internal parameters of the regularization term."""
        pass

    @abc.abstractmethod
    def compute_objective(self) -> float:
        """Computes the objective value corresponding to the regularization term.

        Returns:
            The objective value of the term.

        """
        pass

    @abc.abstractmethod
    def scale(self) -> None:
        """Scales the regularization term."""
        pass


class VolumeRegularization(ShapeRegularizationTerm):
    """Quadratic volume regularization."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        super().__init__(db)

        self.mu = self.config.getfloat("Regularization", "factor_volume")
        self.target_volume = self.config.getfloat("Regularization", "target_volume")
        if self.config.getboolean("Regularization", "use_initial_volume"):
            self.target_volume = self._compute_volume()

        if self.mu > 0.0:
            self.is_active = True

        self.scale()
        self.current_volume = fenics.Expression("val", degree=0, val=1.0)

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the shape derivative of the regularization term.

        Returns:
            The ufl form of the shape derivative.

        """
        if self.is_active:
            shape_form = (
                fenics.Constant(self.mu)
                * (self.current_volume - fenics.Constant(self.target_volume))
                * fenics.div(self.test_vector_field)
                * self.dx
            )
            return shape_form
        else:
            return fenics.derivative(
                fenics.Constant(0.0) * self.dx,
                fenics.SpatialCoordinate(self.mesh),
                self.test_vector_field,
            )

    def update(self) -> None:
        """Updates the internal parameters of the regularization term."""
        self.current_volume.val = self._compute_volume()

    def compute_objective(self) -> float:
        """Computes the objective value corresponding to the regularization term.

        Returns:
            The objective value of the term.

        """
        value = 0.0
        if self.is_active:
            volume = self._compute_volume()
            value += 0.5 * self.mu * pow(volume - self.target_volume, 2)

        return value

    def scale(self) -> None:
        """Scales the regularization term."""
        if self.use_relative_scaling and self.is_active:
            if not self.db.parameter_db.temp_dict:
                volume = self._compute_volume()
                value = 0.5 * pow(volume - self.target_volume, 2)

                if abs(value) < 1e-15:
                    _loggers.info(
                        "The volume regularization vanishes for the initial "
                        "iteration. Multiplying this term with the factor you "
                        "supplied as weight."
                    )
                else:
                    self.mu /= abs(value)

            else:
                self.mu = self.db.parameter_db.temp_dict["Regularization"]["mu_volume"]

    def _compute_volume(self) -> float:
        """Computes the volume of the geometry.

        Returns:
            The volume of the geometry.

        """
        volume: float = fenics.assemble(fenics.Constant(1.0) * self.dx)
        return volume


class SurfaceRegularization(ShapeRegularizationTerm):
    """Quadratic surface regularization."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        super().__init__(db)

        self.ds = fenics.Measure("ds", self.mesh)
        self.mu = self.config.getfloat("Regularization", "factor_surface")
        self.target_surface = self.config.getfloat("Regularization", "target_surface")
        if self.config.getboolean("Regularization", "use_initial_surface"):
            self.target_surface = self._compute_surface()

        if self.mu > 0.0:
            self.is_active = True

        self.scale()
        self.current_surface = fenics.Expression("val", degree=0, val=1.0)

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the shape derivative of the regularization term.

        Returns:
            The ufl form of the shape derivative.

        """
        if self.is_active:
            n = fenics.FacetNormal(self.mesh)
            shape_form = (
                fenics.Constant(self.mu)
                * (self.current_surface - fenics.Constant(self.target_surface))
                * t_div(self.test_vector_field, n)
                * self.ds
            )
            return shape_form
        else:
            return fenics.derivative(
                fenics.Constant(0.0) * self.dx,
                fenics.SpatialCoordinate(self.mesh),
                self.test_vector_field,
            )

    def update(self) -> None:
        """Updates the internal parameters of the regularization term."""
        self.current_surface.val = self._compute_surface()

    def compute_objective(self) -> float:
        """Computes the objective value corresponding to the regularization term.

        Returns:
            The objective value of the term.

        """
        value = 0.0
        if self.is_active:
            surface = self._compute_surface()
            value = 0.5 * self.mu * pow(surface - self.target_surface, 2)

        return value

    def scale(self) -> None:
        """Scales the regularization term."""
        if self.use_relative_scaling and self.is_active:
            if not self.db.parameter_db.temp_dict:
                surface = fenics.assemble(fenics.Constant(1.0) * self.ds)
                value = 0.5 * pow(surface - self.target_surface, 2)

                if abs(value) < 1e-15:
                    _loggers.info(
                        "The surface regularization vanishes for the initial "
                        "iteration. Multiplying this term with the factor you "
                        "supplied as weight."
                    )
                else:
                    self.mu /= abs(value)
            else:
                self.mu = self.db.parameter_db.temp_dict["Regularization"]["mu_surface"]

    def _compute_surface(self) -> float:
        """Computes the surface of the geometry.

        Returns:
            The surface of the geometry.

        """
        surface: float = fenics.assemble(fenics.Constant(1) * self.ds)
        return surface


class BarycenterRegularization(ShapeRegularizationTerm):
    """Quadratic barycenter regularization term."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        super().__init__(db)

        self.geometric_dimension = db.geometry_db.mesh.geometric_dimension()
        self.spatial_coordinate = fenics.SpatialCoordinate(self.mesh)

        self.mu = self.config.getfloat("Regularization", "factor_barycenter")
        self.target_barycenter_list = self.config.getlist(
            "Regularization", "target_barycenter"
        )
        if self.geometric_dimension == 2 and len(self.target_barycenter_list) == 2:
            self.target_barycenter_list.append(0.0)

        if self.config.getboolean("Regularization", "use_initial_barycenter"):
            self.target_barycenter_list = self._compute_barycenter_list()

        if self.mu > 0.0:
            self.is_active = True

        self.scale()

        self.current_barycenter_x = fenics.Expression("val", degree=0, val=0.0)
        self.current_barycenter_y = fenics.Expression("val", degree=0, val=0.0)
        self.current_barycenter_z = fenics.Expression("val", degree=0, val=0.0)
        self.current_volume = fenics.Expression("val", degree=0, val=1.0)

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the shape derivative of the regularization term.

        Returns:
            The ufl form of the shape derivative.

        """
        if self.is_active:
            shape_form = (
                fenics.Constant(self.mu)
                * (
                    self.current_barycenter_x
                    - fenics.Constant(self.target_barycenter_list[0])
                )
                * (
                    self.current_barycenter_x
                    / self.current_volume
                    * fenics.div(self.test_vector_field)
                    + 1
                    / self.current_volume
                    * (
                        self.test_vector_field[0]
                        + self.spatial_coordinate[0]
                        * fenics.div(self.test_vector_field)
                    )
                )
                * self.dx
                + fenics.Constant(self.mu)
                * (
                    self.current_barycenter_y
                    - fenics.Constant(self.target_barycenter_list[1])
                )
                * (
                    self.current_barycenter_y
                    / self.current_volume
                    * fenics.div(self.test_vector_field)
                    + 1
                    / self.current_volume
                    * (
                        self.test_vector_field[1]
                        + self.spatial_coordinate[1]
                        * fenics.div(self.test_vector_field)
                    )
                )
                * self.dx
            )

            if self.geometric_dimension == 3:
                shape_form += (
                    fenics.Constant(self.mu)
                    * (
                        self.current_barycenter_z
                        - fenics.Constant(self.target_barycenter_list[2])
                    )
                    * (
                        self.current_barycenter_z
                        / self.current_volume
                        * fenics.div(self.test_vector_field)
                        + 1
                        / self.current_volume
                        * (
                            self.test_vector_field[2]
                            + self.spatial_coordinate[2]
                            * fenics.div(self.test_vector_field)
                        )
                    )
                    * self.dx
                )
            return shape_form
        else:
            return fenics.derivative(
                fenics.Constant(0.0) * self.dx,
                fenics.SpatialCoordinate(self.mesh),
                self.test_vector_field,
            )

    def update(self) -> None:
        """Updates the internal parameters of the regularization term."""
        barycenter_list = self._compute_barycenter_list()
        volume = fenics.assemble(fenics.Constant(1.0) * self.dx)

        self.current_barycenter_x.val = barycenter_list[0]
        self.current_barycenter_y.val = barycenter_list[1]
        self.current_barycenter_z.val = barycenter_list[2]
        self.current_volume.val = volume

    def compute_objective(self) -> float:
        """Computes the objective value corresponding to the regularization term.

        Returns:
            The objective value of the term.

        """
        value = 0.0

        if self.is_active:
            barycenter_list = self._compute_barycenter_list()

            value = (
                0.5
                * self.mu
                * (
                    pow(barycenter_list[0] - self.target_barycenter_list[0], 2)
                    + pow(barycenter_list[1] - self.target_barycenter_list[1], 2)
                    + pow(barycenter_list[2] - self.target_barycenter_list[2], 2)
                )
            )

        return value

    def scale(self) -> None:
        """Scales the regularization term."""
        if self.use_relative_scaling and self.is_active:
            if not self.db.parameter_db.temp_dict:
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
                    self.mu /= abs(value)
            else:
                self.mu = self.db.parameter_db.temp_dict["Regularization"][
                    "mu_barycenter"
                ]

    def _compute_barycenter_list(self) -> List[float]:
        """Computes the list of barycenters for the geometry.

        Returns:
            The list of coordinates of the barycenter of the geometry.

        """
        barycenter_list = [0.0] * 3
        volume = fenics.assemble(fenics.Constant(1.0) * self.dx)

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


class CurvatureRegularization(ShapeRegularizationTerm):
    """Quadratic curvature regularization term."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        super().__init__(db)

        self.geometric_dimension = db.geometry_db.mesh.geometric_dimension()
        self.ds = fenics.Measure("ds", self.mesh)
        self.spatial_coordinate = fenics.SpatialCoordinate(self.mesh)

        self.a_curvature_matrix = fenics.PETScMatrix()
        self.b_curvature = fenics.PETScVector()

        self.mu = self.config.getfloat("Regularization", "factor_curvature")
        self.kappa_curvature = fenics.Function(self.db.function_db.control_spaces[0])
        n = fenics.FacetNormal(self.mesh)
        x = fenics.SpatialCoordinate(self.mesh)
        self.a_curvature = (
            fenics.inner(
                fenics.TrialFunction(self.db.function_db.control_spaces[0]),
                fenics.TestFunction(self.db.function_db.control_spaces[0]),
            )
            * self.ds
        )
        self.l_curvature = (
            fenics.inner(
                t_grad(x, n),
                t_grad(fenics.TestFunction(self.db.function_db.control_spaces[0]), n),
            )
            * self.ds
        )

        if self.mu > 0:
            self.is_active = True

        self.scale()

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the shape derivative of the regularization term.

        Returns:
            The ufl form of the shape derivative.

        """
        if self.is_active:
            x = fenics.SpatialCoordinate(self.mesh)
            n = fenics.FacetNormal(self.mesh)
            identity = fenics.Identity(self.geometric_dimension)

            shape_form = fenics.Constant(self.mu) * (
                fenics.inner(
                    (identity - (t_grad(x, n) + (t_grad(x, n)).T))
                    * t_grad(self.test_vector_field, n),
                    t_grad(self.kappa_curvature, n),
                )
                * self.ds
                + fenics.Constant(0.5)
                * t_div(self.test_vector_field, n)
                * t_div(self.kappa_curvature, n)
                * self.ds
            )
            return shape_form
        else:
            return fenics.derivative(
                fenics.Constant(0.0) * self.dx,
                fenics.SpatialCoordinate(self.mesh),
                self.test_vector_field,
            )

    def compute_objective(self) -> float:
        """Computes the objective value corresponding to the regularization term.

        Returns:
            The objective value of the term.

        """
        value = 0.0

        if self.is_active:
            self._compute_curvature()
            curvature_val = fenics.assemble(
                fenics.inner(self.kappa_curvature, self.kappa_curvature) * self.ds
            )
            value += 0.5 * self.mu * curvature_val

        return value

    def _compute_curvature(self) -> None:
        """Computes the curvature of the geometry."""
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

    def scale(self) -> None:
        """Scales the regularization term."""
        if self.use_relative_scaling and self.is_active:
            if not self.db.parameter_db.temp_dict:
                self._compute_curvature()
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
                    self.mu /= abs(value)
            else:
                self.mu = self.db.parameter_db.temp_dict["Regularization"][
                    "mu_curvature"
                ]


class ShapeRegularization:
    """Geometric regularization for shape optimization."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        self.volume_regularization = VolumeRegularization(db)
        self.surface_regularization = SurfaceRegularization(db)
        self.barycenter_regularization = BarycenterRegularization(db)
        self.curvature_regularization = CurvatureRegularization(db)

        self.regularization_list = [
            self.volume_regularization,
            self.surface_regularization,
            self.barycenter_regularization,
            self.curvature_regularization,
        ]

    def update_geometric_quantities(self) -> None:
        """Updates the internal parameters of the regularization terms."""
        for term in self.regularization_list:
            term.update()

    def compute_objective(self) -> float:
        """Computes the objective value corresponding to the regularization term.

        Returns:
            The objective value of the term.

        """
        value = 0.0
        for term in self.regularization_list:
            value += term.compute_objective()

        return value

    def compute_shape_derivative(self) -> ufl.Form:
        """Computes the shape derivative of the regularization term.

        Returns:
            The ufl form of the shape derivative.

        """
        shape_derivative_list = [
            term.compute_shape_derivative() for term in self.regularization_list
        ]
        return _utils.summation(shape_derivative_list)
