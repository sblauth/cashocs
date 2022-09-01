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

"""Module for managing UFL forms for shape optimization problems."""

from __future__ import annotations

import itertools
from typing import List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
import ufl
import ufl.algorithms

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils
from cashocs._forms import form_handler
from cashocs._forms import shape_regularization
from cashocs.geometry import boundary_distance

if TYPE_CHECKING:
    from cashocs._optimization import shape_optimization


class ShapeFormHandler(form_handler.FormHandler):
    """Derives adjoint equations and shape derivatives.

    This class is used analogously to the ControlFormHandler class, but for
    shape optimization problems, where it is used to derive the adjoint equations
    and the shape derivatives.
    """

    scalar_product_matrix: fenics.PETScMatrix
    material_derivative: ufl.Form

    def __init__(
        self, optimization_problem: shape_optimization.ShapeOptimizationProblem
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding shape optimization problem.

        """
        super().__init__(optimization_problem)

        self.has_cashocs_remesh_flag: bool = (
            optimization_problem.has_cashocs_remesh_flag
        )
        self.temp_dir: Optional[str] = optimization_problem.temp_dir
        self.boundaries = optimization_problem.boundaries
        self.shape_scalar_product = optimization_problem.shape_scalar_product
        self.uses_custom_scalar_product = (
            optimization_problem.uses_custom_scalar_product
        )
        deformation_space = optimization_problem.deformation_space

        self.degree_estimation = self.config.getboolean(
            "ShapeGradient", "degree_estimation"
        )
        self.use_pull_back = self.config.getboolean("ShapeGradient", "use_pull_back")
        self.use_distance_mu = self.config.getboolean(
            "ShapeGradient", "use_distance_mu"
        )
        self.update_inhomogeneous = self.config.getboolean(
            "ShapeGradient", "update_inhomogeneous"
        )

        self.deformation_space = deformation_space or fenics.VectorFunctionSpace(
            self.mesh, "CG", 1
        )

        self.control_spaces = [self.deformation_space]
        self.control_dim = 1
        self.require_control_constraints = False

        self.gradient = [fenics.Function(self.deformation_space)]
        self.test_vector_field = fenics.TestFunction(self.deformation_space)

        self.shape_regularization: shape_regularization.ShapeRegularization = (
            shape_regularization.ShapeRegularization(self)
        )

        fixed_dimensions = self.config.getlist("ShapeGradient", "fixed_dimensions")
        self.use_fixed_dimensions = False
        if len(fixed_dimensions) > 0:
            self.use_fixed_dimensions = True
            unpack_list = [
                self.deformation_space.sub(i).dofmap().dofs() for i in fixed_dimensions
            ]
            self.fixed_indices = list(itertools.chain(*unpack_list))

        # Calculate the necessary UFL forms
        self.inhomogeneous_mu = False
        self._compute_shape_derivative()
        self._compute_shape_gradient_forms()
        self._setup_mu_computation()

        self.setup_assembler(
            self.riesz_scalar_product, self.shape_derivative, self.bcs_shape
        )
        self.fe_scalar_product_matrix = fenics.PETScMatrix()
        self.fe_shape_derivative_vector = fenics.PETScVector()

        self.A_mu_matrix = fenics.PETScMatrix()  # pylint: disable=invalid-name
        self.b_mu = fenics.PETScVector()

        self.update_scalar_product()
        self._compute_p_laplacian_forms()

        # test for symmetry
        if not self.scalar_product_matrix.isSymmetric():
            if not self.scalar_product_matrix.isSymmetric(1e-15):
                if (
                    not (
                        self.scalar_product_matrix
                        - self.scalar_product_matrix.copy().transpose()
                    ).norm()
                    / self.scalar_product_matrix.norm()
                    < 1e-15
                ):
                    raise _exceptions.InputError(
                        "cashocs._forms.ShapeFormHandler",
                        "shape_scalar_product",
                        "Supplied scalar product form is not symmetric.",
                    )

        if self.opt_algo.casefold() == "newton":
            raise NotImplementedError(
                "Second order methods are not implemented for shape optimization yet"
            )

    def setup_assembler(
        self,
        scalar_product: ufl.form,
        shape_derivative: ufl.form,
        bcs: Optional[List[fenics.DirichletBC]],
    ) -> None:
        """Sets up the assembler for assembling the shape gradient projection.

        Args:
            scalar_product: The weak form of the scalar product
            shape_derivative: The weak form of the shape derivative
            bcs: The boundary conditions for the projection

        """
        modified_scalar_product = _utils.bilinear_boundary_form_modification(
            [scalar_product]
        )[0]
        self.modified_scalar_product = _utils.bilinear_boundary_form_modification(
            [scalar_product]
        )[0]
        retry_assembler_setup = False
        if not self.degree_estimation:
            try:
                self.assembler = fenics.SystemAssembler(
                    modified_scalar_product, shape_derivative, bcs
                )
            except (AssertionError, ValueError):
                retry_assembler_setup = True

        if retry_assembler_setup or self.degree_estimation:
            estimated_degree = np.maximum(
                ufl.algorithms.estimate_total_polynomial_degree(
                    modified_scalar_product
                ),
                ufl.algorithms.estimate_total_polynomial_degree(shape_derivative),
            )
            self.assembler = fenics.SystemAssembler(
                modified_scalar_product,
                shape_derivative,
                bcs,
                form_compiler_parameters={"quadrature_degree": estimated_degree},
            )
        self.assembler.keep_diagonal = True

    def _check_coefficient_id(self, coeff: ufl.core.expr.Expr) -> None:
        """Checks, whether the coefficient belongs to state or adjoint variables.

        Args:
            coeff: The coefficient under investigation

        """
        if (
            coeff.id() not in self.state_adjoint_ids
            and not coeff.ufl_element().family() == "Real"
        ):
            self.material_derivative_coeffs.append(coeff)

    def _parse_pull_back_coefficients(self) -> None:
        """Parses the coefficients which are available for adding pullbacks."""
        self.state_adjoint_ids: List[int] = [coeff.id() for coeff in self.states] + [
            coeff.id() for coeff in self.adjoints
        ]

        self.material_derivative_coeffs: List[ufl.core.expr.Expr] = []

        for coeff in self.lagrangian.coefficients():
            self._check_coefficient_id(coeff)

        if len(self.material_derivative_coeffs) > 0:
            _loggers.warning(
                "Shape derivative might be wrong, if differential operators "
                "act on variables other than states and adjoints."
            )

    def _add_pull_backs(self) -> None:
        """Add pullbacks to the shape derivative."""
        if self.use_pull_back:
            self._parse_pull_back_coefficients()

            for coeff in self.material_derivative_coeffs:

                self.material_derivative = self.lagrangian.derivative(
                    coeff, fenics.dot(fenics.grad(coeff), self.test_vector_field)
                )

                self.material_derivative = ufl.algorithms.expand_derivatives(
                    self.material_derivative
                )

                self.shape_derivative += self.material_derivative

    def _compute_shape_derivative(self) -> None:
        """Calculates the shape derivative.

        This only works properly if differential operators only
        act on state and adjoint variables, else the results are incorrect.
        A corresponding warning whenever this could be the case is issued.
        """
        # Shape derivative of Lagrangian w/o regularization and pullbacks
        self.shape_derivative = self.lagrangian.derivative(
            fenics.SpatialCoordinate(self.mesh), self.test_vector_field
        )

        self._add_pull_backs()

        # Add regularization
        self.shape_derivative += self.shape_regularization.compute_shape_derivative()

    def _compute_shape_gradient_forms(self) -> None:
        """Calculates the necessary left-hand-sides for the shape gradient problem."""
        self.shape_bdry_def = self.config.getlist("ShapeGradient", "shape_bdry_def")
        self.shape_bdry_fix = self.config.getlist("ShapeGradient", "shape_bdry_fix")

        self.shape_bdry_fix_x = self.config.getlist("ShapeGradient", "shape_bdry_fix_x")
        self.shape_bdry_fix_y = self.config.getlist("ShapeGradient", "shape_bdry_fix_y")
        self.shape_bdry_fix_z = self.config.getlist("ShapeGradient", "shape_bdry_fix_z")

        self.bcs_shape = _utils.create_dirichlet_bcs(
            self.deformation_space,
            fenics.Constant([0] * self.deformation_space.ufl_element().value_size()),
            self.boundaries,
            self.shape_bdry_fix,
        )
        self.bcs_shape += _utils.create_dirichlet_bcs(
            self.deformation_space.sub(0),
            fenics.Constant(0.0),
            self.boundaries,
            self.shape_bdry_fix_x,
        )
        self.bcs_shape += _utils.create_dirichlet_bcs(
            self.deformation_space.sub(1),
            fenics.Constant(0.0),
            self.boundaries,
            self.shape_bdry_fix_y,
        )
        if self.deformation_space.num_sub_spaces() == 3:
            self.bcs_shape += _utils.create_dirichlet_bcs(
                self.deformation_space.sub(2),
                fenics.Constant(0.0),
                self.boundaries,
                self.shape_bdry_fix_z,
            )

        self.cg_function_space = fenics.FunctionSpace(self.mesh, "CG", 1)
        self.dg_function_space = fenics.FunctionSpace(self.mesh, "DG", 0)
        self.volumes = fenics.Function(self.dg_function_space)

        self.mu_lame: fenics.Function = fenics.Function(self.cg_function_space)
        self.mu_lame.vector().vec().set(1.0)
        self.mu_lame.vector().apply("")

        if self.shape_scalar_product is None:
            # Use the default linear elasticity approach

            self.lambda_lame = self.config.getfloat("ShapeGradient", "lambda_lame")
            self.damping_factor = self.config.getfloat(
                "ShapeGradient", "damping_factor"
            )

            if self.config.getboolean("ShapeGradient", "inhomogeneous"):
                self.volumes.vector().vec().aypx(
                    0.0,
                    fenics.project(fenics.CellVolume(self.mesh), self.dg_function_space)
                    .vector()
                    .vec(),
                )
                self.volumes.vector().apply("")

                vol_max = self.volumes.vector().max()
                self.volumes.vector().vec().scale(1 / vol_max)
                self.volumes.vector().apply("")

            else:
                self.volumes = fenics.Constant(1.0)

            def eps(u: fenics.Function) -> ufl.core.expr.Expr:
                """Computes the symmetric gradient of a vector field ``u``.

                Args:
                    u: A vector field

                Returns:
                    The symmetric gradient of ``u``

                """
                return fenics.Constant(0.5) * (fenics.grad(u) + fenics.grad(u).T)

            trial = fenics.TrialFunction(self.deformation_space)
            test = fenics.TestFunction(self.deformation_space)

            self.riesz_scalar_product: ufl.Form = (
                fenics.Constant(2)
                * self.mu_lame
                / self.volumes
                * fenics.inner(eps(trial), eps(test))
                * self.dx
                + fenics.Constant(self.lambda_lame)
                / self.volumes
                * fenics.div(trial)
                * fenics.div(test)
                * self.dx
                + fenics.Constant(self.damping_factor)
                / self.volumes
                * fenics.inner(trial, test)
                * self.dx
            )

        else:
            # Use the scalar product supplied by the user
            self.riesz_scalar_product = self.shape_scalar_product

    def _setup_mu_computation(self) -> None:
        """Sets up the computation of the elasticity parameter mu."""
        if not self.use_distance_mu:
            self.mu_def = self.config.getfloat("ShapeGradient", "mu_def")
            self.mu_fix = self.config.getfloat("ShapeGradient", "mu_fix")

            if np.abs(self.mu_def - self.mu_fix) / self.mu_fix > 1e-2:

                self.inhomogeneous_mu = True

                self.options_mu: List[List[Union[str, int, float]]] = [
                    ["ksp_type", "cg"],
                    ["pc_type", "hypre"],
                    ["pc_hypre_type", "boomeramg"],
                    ["ksp_rtol", 1e-16],
                    ["ksp_atol", 1e-50],
                    ["ksp_max_it", 100],
                ]

                phi = fenics.TrialFunction(self.cg_function_space)
                psi = fenics.TestFunction(self.cg_function_space)

                # pylint: disable=invalid-name
                self.A_mu = fenics.inner(fenics.grad(phi), fenics.grad(psi)) * self.dx
                self.l_mu = fenics.Constant(0.0) * psi * self.dx

                self.bcs_mu = _utils.create_dirichlet_bcs(
                    self.cg_function_space,
                    fenics.Constant(self.mu_fix),
                    self.boundaries,
                    self.shape_bdry_fix,
                )
                self.bcs_mu += _utils.create_dirichlet_bcs(
                    self.cg_function_space,
                    fenics.Constant(self.mu_def),
                    self.boundaries,
                    self.shape_bdry_def,
                )

        else:
            self.mu_min = self.config.getfloat("ShapeGradient", "mu_min")
            self.mu_max = self.config.getfloat("ShapeGradient", "mu_max")

            if np.abs(self.mu_min - self.mu_max) / self.mu_min > 1e-2:
                self.dist_min = self.config.getfloat("ShapeGradient", "dist_min")
                self.dist_max = self.config.getfloat("ShapeGradient", "dist_max")

                self.bdry_idcs = self.config.getlist("ShapeGradient", "boundaries_dist")

                self.smooth_mu = self.config.getboolean("ShapeGradient", "smooth_mu")
                self.distance = fenics.Function(self.cg_function_space)
                if not self.smooth_mu:
                    self.mu_expression = fenics.Expression(
                        (
                            "(dist <= dist_min) ? mu_min : "
                            "(dist <= dist_max) ? mu_min + (dist - dist_min)/"
                            "(dist_max - dist_min)*(mu_max - mu_min) : mu_max"
                        ),
                        degree=1,
                        dist=self.distance,
                        dist_min=self.dist_min,
                        dist_max=self.dist_max,
                        mu_min=self.mu_min,
                        mu_max=self.mu_max,
                    )
                else:
                    self.mu_expression = fenics.Expression(
                        (
                            "(dist <= dist_min) ? mu_min :"
                            "(dist <= dist_max) ? mu_min + "
                            "(mu_max - mu_min)/(dist_max - dist_min)*(dist - dist_min) "
                            "- (mu_max - mu_min)/pow(dist_max - dist_min, 2)"
                            "*(dist - dist_min)*(dist - dist_max) "
                            "- 2*(mu_max - mu_min)/pow(dist_max - dist_min, 3)"
                            "*(dist - dist_min)*pow(dist - dist_max, 2)"
                            " : mu_max"
                        ),
                        degree=3,
                        dist=self.distance,
                        dist_min=self.dist_min,
                        dist_max=self.dist_max,
                        mu_min=self.mu_min,
                        mu_max=self.mu_max,
                    )

    def _compute_mu_elas(self) -> None:
        """Computes the elasticity parameter mu.

        Based on `Schulz and Siebenborn, Computational Comparison of Surface Metrics for
        PDE Constrained Shape Optimization <https://doi.org/10.1515/cmam-2016-0009>`_.
        """
        if self.shape_scalar_product is None:
            if not self.use_distance_mu:
                if self.inhomogeneous_mu:
                    x = _utils.assemble_and_solve_linear(
                        self.A_mu,
                        self.l_mu,
                        self.bcs_mu,
                        A=self.A_mu_matrix,
                        b=self.b_mu,
                        ksp_options=self.options_mu,
                    )

                    if self.config.getboolean("ShapeGradient", "use_sqrt_mu"):
                        x.sqrtabs()

                    self.mu_lame.vector().vec().aypx(0.0, x)
                    self.mu_lame.vector().apply("")

                else:
                    self.mu_lame.vector().vec().set(self.mu_fix)
                    self.mu_lame.vector().apply("")

            else:
                self.distance.vector().vec().aypx(
                    0.0,
                    boundary_distance.compute_boundary_distance(
                        self.mesh, self.boundaries, self.bdry_idcs
                    )
                    .vector()
                    .vec(),
                )
                self.distance.vector().apply("")
                self.mu_lame.vector().vec().aypx(
                    0.0,
                    fenics.interpolate(self.mu_expression, self.cg_function_space)
                    .vector()
                    .vec(),
                )
                self.mu_lame.vector().apply("")

    def _project_scalar_product(self) -> None:
        """Ensures, that only free dimensions can be deformed."""
        if self.use_fixed_dimensions:

            copy_mat = self.fe_scalar_product_matrix.copy()

            copy_mat.ident(self.fixed_indices)
            copy_mat.mat().transpose()
            copy_mat.ident(self.fixed_indices)
            copy_mat.mat().transpose()

            self.fe_scalar_product_matrix.mat().aypx(0.0, copy_mat.mat())

    def update_scalar_product(self) -> None:
        """Updates the linear elasticity equations to the current geometry.

        Updates the left-hand-side of the linear elasticity equations
        (needed when the geometry changes).
        """
        self._compute_mu_elas()
        if self.update_inhomogeneous:
            self.volumes.vector().vec().aypx(
                0.0,
                fenics.project(fenics.CellVolume(self.mesh), self.dg_function_space)
                .vector()
                .vec(),
            )
            self.volumes.vector().apply("")
            vol_max = self.volumes.vector().vec().max()[1]
            self.volumes.vector().vec().scale(1 / vol_max)
            self.volumes.vector().apply("")

        self.assembler.assemble(self.fe_scalar_product_matrix)
        self.fe_scalar_product_matrix.ident_zeros()
        self.scalar_product_matrix = self.fe_scalar_product_matrix.mat()
        self._project_scalar_product()

    def scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between two deformation functions a and b.

        Args:
            a: The first argument.
            b: The second argument.

        Returns:
            The scalar product of a and b.

        """
        result: float
        if (
            self.config.getboolean("ShapeGradient", "use_p_laplacian")
            and not self.uses_custom_scalar_product
        ):
            form = ufl.replace(
                self.p_laplace_form,
                {self.gradient[0]: a[0], self.test_vector_field: b[0]},
            )
            result = fenics.assemble(form)

        else:

            x = fenics.as_backend_type(a[0].vector()).vec()
            y = fenics.as_backend_type(b[0].vector()).vec()

            temp, _ = self.scalar_product_matrix.getVecs()
            self.scalar_product_matrix.mult(x, temp)
            result = temp.dot(y)

        return result

    def _compute_p_laplacian_forms(self) -> None:
        """Computes the weak forms for the p-Laplace equations."""
        if self.config.getboolean("ShapeGradient", "use_p_laplacian"):
            p = self.config.getint("ShapeGradient", "p_laplacian_power")
            delta = self.config.getfloat("ShapeGradient", "damping_factor")
            eps = self.config.getfloat("ShapeGradient", "p_laplacian_stabilization")
            kappa = pow(
                fenics.inner(
                    fenics.grad(self.gradient[0]), fenics.grad(self.gradient[0])
                ),
                (p - 2) / 2.0,
            )
            self.p_laplace_form: ufl.Form = (
                fenics.inner(
                    self.mu_lame
                    * (fenics.Constant(eps) + kappa)
                    * fenics.grad(self.gradient[0]),
                    fenics.grad(self.test_vector_field),
                )
                * self.dx
                + fenics.Constant(delta)
                * fenics.dot(self.gradient[0], self.test_vector_field)
                * self.dx
            )
