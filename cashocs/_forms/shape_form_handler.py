# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

"""Management of weak forms for shape optimization problems."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import fenics
import numpy as np

try:
    from ufl import algorithms as ufl_algorithms
    import ufl_legacy as ufl
except ImportError:
    import ufl
    from ufl import algorithms as ufl_algorithms

from cashocs import _exceptions
from cashocs import _utils
from cashocs import log
from cashocs._forms import form_handler
from cashocs._forms import shape_regularization
from cashocs.geometry import boundary_distance

if TYPE_CHECKING:
    from petsc4py import PETSc

    try:
        from ufl_legacy.core import expr as ufl_expr
    except ImportError:
        from ufl.core import expr as ufl_expr

    from cashocs import _typing
    from cashocs import io
    from cashocs._database import database
    from cashocs._optimization import shape_optimization


class Stiffness:
    """Stiffness in linear elasticity equations for projecting the shape derivative."""

    def __init__(
        self,
        mu_lame: fenics.Function,
        config: io.Config,
        mesh: fenics.Mesh,
        boundaries: fenics.MeshFunction,
        shape_bdry_def: list[int | str],
        shape_bdry_fix: list[int | str],
    ) -> None:
        """Class for managing the stiffness parameter for shape optimization.

        Args:
            mu_lame: Function representing the stiffness
            config: The config file for the problem
            mesh: The underlying mesh
            boundaries: The boundaries of the mesh
            shape_bdry_def: list of indices of the deformable boundary
            shape_bdry_fix: list of indices of the fixed boundary

        """
        self.mu_lame = mu_lame
        self.config = config
        self.mesh = mesh
        self.boundaries = boundaries
        self.shape_bdry_def = shape_bdry_def
        self.shape_bdry_fix = shape_bdry_fix

        self.comm = self.mesh.mpi_comm()

        self.inhomogeneous_mu = False

        self.dx = ufl.Measure("dx", self.mesh)

        self.use_distance_mu = self.config.getboolean(
            "ShapeGradient", "use_distance_mu"
        )
        self.cg_function_space = self.mu_lame.function_space()
        self.distance = fenics.Function(self.cg_function_space)

        self.A_mu_matrix = fenics.PETScMatrix(self.comm)  # pylint: disable=invalid-name
        self.b_mu = fenics.PETScVector(self.comm)
        self.options_mu: _typing.KspOption = {}

        self._setup_mu_computation()

    def _setup_mu_computation(self) -> None:
        """Sets up the computation of the elasticity parameter mu."""
        if not self.use_distance_mu:
            mu_def = self.config.getfloat("ShapeGradient", "mu_def")
            mu_fix = self.config.getfloat("ShapeGradient", "mu_fix")

            if np.abs(mu_def - mu_fix) / mu_fix > 1e-2:
                self.inhomogeneous_mu = True

                self.options_mu = {
                    "ksp_type": "cg",
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "ksp_rtol": 1e-16,
                    "ksp_atol": 1e-50,
                    "ksp_max_it": 100,
                }

                phi = fenics.TrialFunction(self.cg_function_space)
                psi = fenics.TestFunction(self.cg_function_space)

                # pylint: disable=invalid-name
                self.A_mu = ufl.inner(ufl.grad(phi), ufl.grad(psi)) * self.dx
                self.l_mu = fenics.Constant(0.0) * psi * self.dx

                self.bcs_mu = _utils.create_dirichlet_bcs(
                    self.cg_function_space,
                    fenics.Constant(mu_fix),
                    self.boundaries,
                    self.shape_bdry_fix,
                )
                self.bcs_mu += _utils.create_dirichlet_bcs(
                    self.cg_function_space,
                    fenics.Constant(mu_def),
                    self.boundaries,
                    self.shape_bdry_def,
                )

        else:
            mu_min = self.config.getfloat("ShapeGradient", "mu_min")
            mu_max = self.config.getfloat("ShapeGradient", "mu_max")

            if np.abs(mu_min - mu_max) / mu_min > 1e-2:
                dist_min = self.config.getfloat("ShapeGradient", "dist_min")
                dist_max = self.config.getfloat("ShapeGradient", "dist_max")

                self.bdry_idcs = self.config.getlist("ShapeGradient", "boundaries_dist")

                smooth_mu = self.config.getboolean("ShapeGradient", "smooth_mu")

                if not smooth_mu:
                    self.mu_expression = fenics.Expression(
                        (
                            "(dist <= dist_min) ? mu_min : "
                            "(dist <= dist_max) ? mu_min + (dist - dist_min)/"
                            "(dist_max - dist_min)*(mu_max - mu_min) : mu_max"
                        ),
                        degree=1,
                        dist=self.distance,
                        dist_min=dist_min,
                        dist_max=dist_max,
                        mu_min=mu_min,
                        mu_max=mu_max,
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
                        dist_min=dist_min,
                        dist_max=dist_max,
                        mu_min=mu_min,
                        mu_max=mu_max,
                    )

    def compute(self) -> None:
        """Computes the elasticity parameter mu.

        Based on `Schulz and Siebenborn, Computational Comparison of Surface Metrics for
        PDE Constrained Shape Optimization <https://doi.org/10.1515/cmam-2016-0009>`_.
        """
        if not self.use_distance_mu:
            if self.inhomogeneous_mu:
                _utils.assemble_and_solve_linear(
                    self.A_mu,
                    self.l_mu,
                    self.mu_lame,
                    bcs=self.bcs_mu,
                    A=self.A_mu_matrix,
                    b=self.b_mu,
                    ksp_options=self.options_mu,
                )

                if self.config.getboolean("ShapeGradient", "use_sqrt_mu"):
                    self.mu_lame.sqrtabs()
                    self.mu_lame.vector().apply("")

            else:
                self.mu_lame.vector().vec().set(
                    self.config.getfloat("ShapeGradient", "mu_fix")
                )
                self.mu_lame.vector().apply("")

        else:
            self.distance.vector().vec().aypx(
                0.0,
                boundary_distance.compute_boundary_distance(
                    self.mesh,
                    self.boundaries,
                    self.bdry_idcs,
                    method=self.config.get("ShapeGradient", "distance_method"),
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


class ShapeFormHandler(form_handler.FormHandler):
    """Derives adjoint equations and shape derivatives.

    This class is used analogously to the ControlFormHandler class, but for
    shape optimization problems, where it is used to derive the adjoint equations
    and the shape derivatives.
    """

    use_fixed_dimensions: bool
    bcs_shape: list[fenics.DirichletBC]
    fe_shape_derivative_vector: fenics.PETScVector
    shape_derivative: ufl.Form
    fixed_indices: list[int]
    assembler: fenics.SystemAssembler
    assembler_extension: fenics.SystemAssembler
    scalar_product_matrix: fenics.PETScMatrix
    modified_scalar_product: ufl.Form

    def __init__(
        self,
        optimization_problem: shape_optimization.ShapeOptimizationProblem,
        db: database.Database,
        regularization: shape_regularization.ShapeRegularization,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding shape optimization problem.
            db: The database of the problem.
            regularization: The geometric regularization of the problem.

        """
        super().__init__(optimization_problem, db)

        self.boundaries = optimization_problem.boundaries
        self.shape_scalar_product = optimization_problem.shape_scalar_product
        self.uses_custom_scalar_product = (
            optimization_problem.uses_custom_scalar_product
        )

        self.degree_estimation = self.config.getboolean(
            "ShapeGradient", "degree_estimation"
        )
        self.use_pull_back = self.config.getboolean("ShapeGradient", "use_pull_back")
        self.update_inhomogeneous = self.config.getboolean(
            "ShapeGradient", "update_inhomogeneous"
        )

        self.shape_bdry_def = self.config.getlist("ShapeGradient", "shape_bdry_def")
        self.shape_bdry_fix = self.config.getlist("ShapeGradient", "shape_bdry_fix")

        self.shape_bdry_fix_x = self.config.getlist("ShapeGradient", "shape_bdry_fix_x")
        self.shape_bdry_fix_y = self.config.getlist("ShapeGradient", "shape_bdry_fix_y")
        self.shape_bdry_fix_z = self.config.getlist("ShapeGradient", "shape_bdry_fix_z")

        self.cg_function_space = fenics.FunctionSpace(self.db.geometry_db.mesh, "CG", 1)
        self.dg_function_space = fenics.FunctionSpace(self.db.geometry_db.mesh, "DG", 0)
        self.mu_lame: fenics.Function = fenics.Function(self.cg_function_space)
        self.volumes = fenics.Function(self.dg_function_space)

        self.stiffness = Stiffness(
            self.mu_lame,
            self.config,
            self.db.geometry_db.mesh,
            self.boundaries,
            self.shape_bdry_def,
            self.shape_bdry_fix,
        )

        self.test_vector_field: fenics.TestFunction = fenics.TestFunction(
            self.db.function_db.control_spaces[0]
        )

        self.shape_regularization: shape_regularization.ShapeRegularization = (
            regularization
        )

        fixed_dimensions = self.config.getlist("ShapeGradient", "fixed_dimensions")
        self.use_fixed_dimensions = False
        if len(fixed_dimensions) > 0:
            self.use_fixed_dimensions = True
            unpack_list = [
                self.db.function_db.control_spaces[0].sub(i).dofmap().dofs()
                for i in fixed_dimensions
            ]
            self.fixed_indices = list(itertools.chain(*unpack_list))

        self.state_adjoint_ids: list[int] = []
        self.material_derivative_coeffs: list[ufl_expr.Expr] = []

        # Calculate the necessary UFL forms
        self.shape_derivative = self._compute_shape_derivative()

        self.bcs_shape = self._setup_bcs_shape()
        self.riesz_scalar_product: ufl.Form = self._compute_shape_gradient_forms()

        self.modified_scalar_product, self.assembler = self.setup_assembler(
            self.riesz_scalar_product, self.shape_derivative, self.bcs_shape
        )

        self.fe_scalar_product_matrix = fenics.PETScMatrix(self.db.geometry_db.mpi_comm)
        self.scalar_product_matrix = self.fe_scalar_product_matrix.mat()
        self.fe_shape_derivative_vector = fenics.PETScVector(
            self.db.geometry_db.mpi_comm
        )

        if self.config.getboolean("ShapeGradient", "reextend_from_boundary"):
            self.bcs_extension = self._setup_bcs_extension()
            zero_source = (
                ufl.dot(
                    self.test_vector_field,
                    fenics.Constant(
                        [0]
                        * self.db.function_db.control_spaces[0]
                        .ufl_element()
                        .value_size()
                    ),
                )
                * self.dx
            )
            _, self.assembler_extension = self.setup_assembler(
                self.riesz_scalar_product, zero_source, self.bcs_extension
            )
            self.fe_reextension_matrix = fenics.PETScMatrix(
                self.db.geometry_db.mpi_comm
            )
            self.reextension_matrix: PETSc.Mat = self.fe_reextension_matrix.mat()
            self.fe_reextension_vector: fenics.PETScVector = fenics.PETScVector(
                self.db.geometry_db.mpi_comm
            )

        self.update_scalar_product()
        self.p_laplace_form = self._compute_p_laplacian_forms()

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
        scalar_product: ufl.Form,
        shape_derivative: ufl.Form,
        bcs: list[fenics.DirichletBC] | None,
    ) -> tuple[ufl.Form, fenics.SystemAssembler]:
        """Sets up the assembler for assembling the shape gradient projection.

        Args:
            scalar_product: The weak form of the scalar product
            shape_derivative: The weak form of the shape derivative
            bcs: The boundary conditions for the projection

        """
        modified_scalar_product = _utils.bilinear_boundary_form_modification(
            [scalar_product]
        )[0]
        if not self.degree_estimation:
            try:
                assembler = fenics.SystemAssembler(
                    modified_scalar_product, shape_derivative, bcs
                )
                assembler.keep_diagonal = True
            except (AssertionError, ValueError):
                assembler = self._setup_assembler_failsafe(
                    modified_scalar_product, shape_derivative, bcs
                )
        else:
            assembler = self._setup_assembler_failsafe(
                modified_scalar_product, shape_derivative, bcs
            )

        return modified_scalar_product, assembler

    def _setup_assembler_failsafe(
        self,
        modified_scalar_product: ufl.Form,
        shape_derivative: ufl.Form,
        bcs: list[fenics.DirichletBC] | None,
    ) -> fenics.SystemAssembler:
        """Set up the assembler in a fail-safe manner.

        Args:
            modified_scalar_product: The weak form of the scalar product
            shape_derivative: The weak form of the shape derivative
            bcs: The boundary conditions for the projection

        Returns:
            The assembler for the system.

        """
        estimated_degree = np.maximum(
            ufl_algorithms.estimate_total_polynomial_degree(modified_scalar_product),
            ufl_algorithms.estimate_total_polynomial_degree(shape_derivative),
        )
        fenics_quadrature_degree = fenics.parameters["form_compiler"][
            "quadrature_degree"
        ]
        if fenics_quadrature_degree is not None:
            estimated_degree = np.minimum(estimated_degree, fenics_quadrature_degree)

        assembler = fenics.SystemAssembler(
            modified_scalar_product,
            shape_derivative,
            bcs,
            form_compiler_parameters={"quadrature_degree": estimated_degree},
        )

        assembler.keep_diagonal = True

        return assembler

    def _check_coefficient_id(self, coeff: ufl_expr.Expr) -> None:
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
        self.state_adjoint_ids = [
            coeff.id() for coeff in self.db.function_db.states
        ] + [coeff.id() for coeff in self.db.function_db.adjoints]

        self.material_derivative_coeffs.clear()

        for coeff in self.lagrangian.coefficients():
            self._check_coefficient_id(coeff)

        if len(self.material_derivative_coeffs) > 0:
            log.warning(
                "Shape derivative might be wrong, if differential operators "
                "act on variables other than states and adjoints."
            )

    def _add_pull_backs(self, shape_derivative: ufl.Form) -> ufl.Form:
        """Add pullbacks to the shape derivative.

        Args:
            shape_derivative: Form of the shape derivative (without pullbacks)

        Returns:
            Form of the shape derivative (with pullbacks)

        """
        if self.use_pull_back:
            self._parse_pull_back_coefficients()

            for coeff in self.material_derivative_coeffs:
                material_derivative = self.lagrangian.derivative(
                    coeff, ufl.dot(ufl.grad(coeff), self.test_vector_field)
                )

                material_derivative = ufl_algorithms.expand_derivatives(
                    material_derivative
                )

                shape_derivative += material_derivative

        return shape_derivative

    def _compute_shape_derivative(self) -> ufl.Form:
        """Calculates the shape derivative.

        This only works properly if differential operators only
        act on state and adjoint variables, else the results are incorrect.
        A corresponding warning whenever this could be the case is issued.
        """
        # Shape derivative of Lagrangian w/o regularization and pullbacks
        shape_derivative = self.lagrangian.derivative(
            fenics.SpatialCoordinate(self.db.geometry_db.mesh), self.test_vector_field
        )

        shape_derivative = self._add_pull_backs(shape_derivative)
        # Add regularization
        shape_derivative += self.shape_regularization.compute_shape_derivative()

        return shape_derivative

    def _setup_bcs_shape(
        self,
    ) -> list[fenics.DirichletBC]:
        """Defines the boundary conditions for the shape deformation.

        Returns:
            The list of boundary conditions

        """
        bcs_shape = _utils.create_dirichlet_bcs(
            self.db.function_db.control_spaces[0],
            fenics.Constant(
                [0] * self.db.function_db.control_spaces[0].ufl_element().value_size()
            ),
            self.boundaries,
            self.shape_bdry_fix,
        )
        bcs_shape += _utils.create_dirichlet_bcs(
            self.db.function_db.control_spaces[0].sub(0),
            fenics.Constant(0.0),
            self.boundaries,
            self.shape_bdry_fix_x,
        )
        bcs_shape += _utils.create_dirichlet_bcs(
            self.db.function_db.control_spaces[0].sub(1),
            fenics.Constant(0.0),
            self.boundaries,
            self.shape_bdry_fix_y,
        )
        if self.db.function_db.control_spaces[0].num_sub_spaces() == 3:
            bcs_shape += _utils.create_dirichlet_bcs(
                self.db.function_db.control_spaces[0].sub(2),
                fenics.Constant(0.0),
                self.boundaries,
                self.shape_bdry_fix_z,
            )

        return bcs_shape

    def _setup_bcs_extension(self) -> list[fenics.DirichletBC]:
        """Defines the DirichletBCs for the re-extensions of the gradient deformation.

        Returns:
            The list of boundary conditions for re-extending the gradient deformation.

        """
        all_boundaries = (
            self.shape_bdry_def
            + self.shape_bdry_fix
            + self.shape_bdry_fix_x
            + self.shape_bdry_fix_y
            + self.shape_bdry_fix_z
        )
        bcs_extension = _utils.create_dirichlet_bcs(
            self.db.function_db.control_spaces[0],
            self.db.function_db.gradient[0],
            self.boundaries,
            all_boundaries,
        )

        return bcs_extension

    def _compute_shape_gradient_forms(self) -> ufl.Form:
        """Calculates the necessary left-hand-sides for the shape gradient problem.

        Returns:
            The left-hand side for the shape gradient problem

        """
        self.mu_lame.vector().vec().set(1.0)
        self.mu_lame.vector().apply("")

        if self.shape_scalar_product is None:
            # Use the default linear elasticity approach

            lambda_lame = self.config.getfloat("ShapeGradient", "lambda_lame")
            damping_factor = self.config.getfloat("ShapeGradient", "damping_factor")

            if self.config.getboolean("ShapeGradient", "inhomogeneous"):
                self.volumes.vector().vec().aypx(
                    0.0,
                    _utils.l2_projection(
                        fenics.CellVolume(self.db.geometry_db.mesh),
                        self.dg_function_space,
                        ksp_options={"ksp_type": "preonly", "pc_type": "jacobi"},
                    )
                    .vector()
                    .vec(),
                )
                self.volumes.vector().apply("")

                vol_max = self.volumes.vector().max()
                self.volumes.vector().vec().scale(1 / vol_max)
                self.volumes.vector().apply("")

                self.inhomogeneous_exponent = fenics.Constant(
                    self.config.getfloat("ShapeGradient", "inhomogeneous_exponent")
                )
            else:
                self.volumes.vector().vec().set(1.0)
                self.volumes.vector().apply("")

                self.inhomogeneous_exponent = fenics.Constant(1.0)

            def eps(u: fenics.Function) -> ufl_expr.Expr:
                """Computes the symmetric gradient of a vector field ``u``.

                Args:
                    u: A vector field

                Returns:
                    The symmetric gradient of ``u``

                """
                return fenics.Constant(0.5) * (ufl.grad(u) + ufl.grad(u).T)

            trial = fenics.TrialFunction(self.db.function_db.control_spaces[0])
            test = fenics.TestFunction(self.db.function_db.control_spaces[0])

            riesz_scalar_product = (
                fenics.Constant(2)
                * self.mu_lame
                / pow(self.volumes, self.inhomogeneous_exponent)
                * ufl.inner(eps(trial), eps(test))
                * self.dx
                + fenics.Constant(lambda_lame)
                / pow(self.volumes, self.inhomogeneous_exponent)
                * ufl.div(trial)
                * ufl.div(test)
                * self.dx
                + fenics.Constant(damping_factor)
                / pow(self.volumes, self.inhomogeneous_exponent)
                * ufl.inner(trial, test)
                * self.dx
            )

        else:
            # Use the scalar product supplied by the user
            riesz_scalar_product = self.shape_scalar_product

        return riesz_scalar_product

    def _project_scalar_product(self) -> None:
        """Ensures, that only free dimensions can be deformed."""
        if self.use_fixed_dimensions:
            copy_mat = self.fe_scalar_product_matrix.copy()

            copy_mat.ident(self.fixed_indices)
            copy_mat.mat().transpose()
            copy_mat.ident(self.fixed_indices)
            copy_mat.mat().transpose()

            self.fe_scalar_product_matrix.mat().aypx(0.0, copy_mat.mat())

            if self.config.getboolean("ShapeGradient", "reextend_from_boundary"):
                copy_mat = self.fe_reextension_matrix.copy()
                copy_mat.ident(self.fixed_indices)
                copy_mat.mat().transpose()
                copy_mat.ident(self.fixed_indices)
                copy_mat.mat().transpose()

                self.fe_reextension_matrix.mat().aypx(0.0, copy_mat.mat())

    def update_scalar_product(self) -> None:
        """Updates the linear elasticity equations to the current geometry.

        Updates the left-hand-side of the linear elasticity equations
        (needed when the geometry changes).
        """
        self._compute_mu_elas()
        if self.update_inhomogeneous:
            self.volumes.vector().vec().aypx(
                0.0,
                _utils.l2_projection(
                    fenics.CellVolume(self.db.geometry_db.mesh),
                    self.dg_function_space,
                    ksp_options={"ksp_type": "preonly", "pc_type": "jacobi"},
                )
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

        if self.config.getboolean("ShapeGradient", "reextend_from_boundary"):
            self.assembler_extension.assemble(self.fe_reextension_matrix)
            self.fe_reextension_matrix.ident_zeros()
            self.reextension_matrix = self.fe_reextension_matrix.mat()

        self._project_scalar_product()

    def scalar_product(
        self, a: list[fenics.Function], b: list[fenics.Function]
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
                {self.db.function_db.gradient[0]: a[0], self.test_vector_field: b[0]},
            )
            result = fenics.assemble(form)

        else:
            x = fenics.as_backend_type(a[0].vector()).vec()
            y = fenics.as_backend_type(b[0].vector()).vec()

            temp = self.scalar_product_matrix.createVecRight()
            self.scalar_product_matrix.mult(x, temp)
            result = temp.dot(y)

        return result

    def _compute_p_laplacian_forms(self) -> ufl.Form:
        """Computes the weak forms for the p-Laplace equations.

        Returns:
            The weak form of the p-Laplace equations

        """
        if self.config.getboolean("ShapeGradient", "use_p_laplacian"):
            p = self.config.getint("ShapeGradient", "p_laplacian_power")
            delta = self.config.getfloat("ShapeGradient", "damping_factor")
            eps = self.config.getfloat("ShapeGradient", "p_laplacian_stabilization")
            kappa = pow(
                ufl.inner(
                    ufl.grad(self.db.function_db.gradient[0]),
                    ufl.grad(self.db.function_db.gradient[0]),
                ),
                (p - 2) / 2.0,
            )
            p_laplace_form = (
                ufl.inner(
                    self.mu_lame
                    * (fenics.Constant(eps) + kappa)
                    * ufl.grad(self.db.function_db.gradient[0]),
                    ufl.grad(self.test_vector_field),
                )
                * self.dx
                + fenics.Constant(delta)
                * ufl.dot(self.db.function_db.gradient[0], self.test_vector_field)
                * self.dx
            )
        else:
            p_laplace_form = fenics.Constant(0) * self.dx

        return p_laplace_form

    def _compute_mu_elas(self) -> None:
        """Computes the elasticity parameter mu.

        Based on `Schulz and Siebenborn, Computational Comparison of Surface Metrics for
        PDE Constrained Shape Optimization <https://doi.org/10.1515/cmam-2016-0009>`_.
        """
        self.stiffness.compute()

    def apply_shape_bcs(self, function: fenics.Function) -> None:
        """Applies the geometric boundary conditions / constraints to a function.

        Args:
            function: The function onto which the geometric constraints are imposed.
                Must be a vector CG1 function.

        """
        for bc in self.bcs_shape:
            bc.apply(function.vector())
            function.vector().apply("")

        if self.use_fixed_dimensions:
            function.vector().vec()[self.fixed_indices] = np.array(
                [0.0] * len(self.fixed_indices)
            )
            function.vector().apply("")

    def apply_reextension_bcs(self, function: fenics.Function) -> None:
        """Applies the boundary conditions for the reextension of the shape gradient.

        Args:
            function (fenics.Function): The function which shall receive the boundary
                values of the shape gradient.

        """
        for bc in self.bcs_extension:
            bc.apply(function.vector())
            function.vector().apply("")

        if self.use_fixed_dimensions:
            function.vector().vec()[self.fixed_indices] = np.array(
                [0.0] * len(self.fixed_indices)
            )
            function.vector().apply("")
