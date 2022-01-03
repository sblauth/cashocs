# Copyright (C) 2020-2022 Sebastian Blauth
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


from __future__ import annotations

import itertools
import json
from typing import List, TYPE_CHECKING

import fenics
import numpy as np
from petsc4py import PETSc
from ufl import replace
from ufl.algorithms import expand_derivatives
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree

from .form_handler import FormHandler
from .._exceptions import (
    InputError,
    IncompatibleConfigurationError,
)
from .._loggers import warning
from .shape_regularization import ShapeRegularization
from ..geometry import compute_boundary_distance
from ..utils import (
    _assemble_petsc_system,
    _setup_petsc_options,
    _solve_linear_problem,
    create_dirichlet_bcs,
    _max,
    _min,
)


if TYPE_CHECKING:
    from .._shape_optimization.shape_optimization_problem import (
        ShapeOptimizationProblem,
    )


class ShapeFormHandler(FormHandler):
    """Derives adjoint equations and shape derivatives.

    This class is used analogously to the ControlFormHandler class, but for
    shape optimization problems, where it is used to derive the adjoint equations
    and the shape derivatives.

    See Also
    --------
    ControlFormHandler : Derives adjoint and gradient equations for optimal control problems
    """

    def __init__(self, optimization_problem: ShapeOptimizationProblem) -> None:
        """Initializes the ShapeFormHandler object.

        Parameters
        ----------
        optimization_problem : ShapeOptimizationProblem
            The corresponding shape optimization problem
        """

        super().__init__(optimization_problem)

        self.has_cashocs_remesh_flag = optimization_problem.has_cashocs_remesh_flag
        self.temp_dir = optimization_problem.temp_dir
        self.boundaries = optimization_problem.boundaries
        self.shape_scalar_product = optimization_problem.shape_scalar_product
        self.uses_custom_scalar_product = (
            optimization_problem.uses_custom_scalar_product
        )
        deformation_space = optimization_problem.deformation_space

        self.control_dim = 1

        self.degree_estimation = self.config.getboolean(
            "ShapeGradient", "degree_estimation", fallback=True
        )
        self.use_pull_back = self.config.getboolean(
            "ShapeGradient", "use_pull_back", fallback=True
        )
        self.use_distance_mu = self.config.getboolean(
            "ShapeGradient", "use_distance_mu", fallback=False
        )
        self.update_inhomogeneous = self.config.getboolean(
            "ShapeGradient", "update_inhomogeneous", fallback=False
        )

        if deformation_space is None:
            self.deformation_space = fenics.VectorFunctionSpace(self.mesh, "CG", 1)
        else:
            self.deformation_space = deformation_space

        self.control_spaces = [self.deformation_space]

        self.gradient = [fenics.Function(self.deformation_space)]
        self.test_vector_field = fenics.TestFunction(self.deformation_space)

        self.shape_regularization = ShapeRegularization(self)

        temp_fixed_dimensions = self.config.get(
            "ShapeGradient", "fixed_dimensions", fallback="[]"
        )
        fixed_dimensions = json.loads(temp_fixed_dimensions)
        self.use_fixed_dimensions = False
        if len(fixed_dimensions) > 0:
            self.use_fixed_dimensions = True
            unpack_list = [
                self.deformation_space.sub(i).dofmap().dofs() for i in fixed_dimensions
            ]
            self.fixed_indices = list(itertools.chain(*unpack_list))

        if (
            self.use_fixed_dimensions
            and self.config.getboolean(
                "ShapeGradient", "use_p_laplacian", fallback=False
            )
            and not self.form_handler.uses_custom_scalar_product
        ):
            raise IncompatibleConfigurationError(
                "use_p_laplacian", "ShapeGradient", "fixed_dimensions", "ShapeGradient"
            )

        # Calculate the necessary UFL forms
        self.inhomogeneous_mu = False
        self.__compute_shape_derivative()
        self.__compute_shape_gradient_forms()
        self.__setup_mu_computation()

        if self.degree_estimation:
            self.estimated_degree = np.maximum(
                estimate_total_polynomial_degree(self.riesz_scalar_product),
                estimate_total_polynomial_degree(self.shape_derivative),
            )
            self.assembler = fenics.SystemAssembler(
                self.riesz_scalar_product,
                self.shape_derivative,
                self.bcs_shape,
                form_compiler_parameters={"quadrature_degree": self.estimated_degree},
            )
        else:
            try:
                self.assembler = fenics.SystemAssembler(
                    self.riesz_scalar_product, self.shape_derivative, self.bcs_shape
                )
            except (AssertionError, ValueError):
                self.estimated_degree = np.maximum(
                    estimate_total_polynomial_degree(self.riesz_scalar_product),
                    estimate_total_polynomial_degree(self.shape_derivative),
                )
                self.assembler = fenics.SystemAssembler(
                    self.riesz_scalar_product,
                    self.shape_derivative,
                    self.bcs_shape,
                    form_compiler_parameters={
                        "quadrature_degree": self.estimated_degree
                    },
                )

        self.assembler.keep_diagonal = True
        self.fe_scalar_product_matrix = fenics.PETScMatrix()
        self.fe_shape_derivative_vector = fenics.PETScVector()

        self.A_mu = fenics.PETScMatrix()
        self.b_mu = fenics.PETScVector()

        self.update_scalar_product()
        self.__compute_p_laplacian_forms()

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
                    raise InputError(
                        "cashocs._forms.ShapeFormHandler",
                        "shape_scalar_product",
                        "Supplied scalar product form is not symmetric.",
                    )

        if self.opt_algo == "newton" or (
            self.opt_algo == "pdas" and self.inner_pdas == "newton"
        ):
            raise NotImplementedError(
                "Second order methods are not implemented for shape optimization yet"
            )

    def __compute_shape_derivative(self) -> None:
        """Computes the shape derivative.

        Returns
        -------
        None

        Notes
        -----
        This only works properly if differential operators only
        act on state and adjoint variables, else the results are incorrect.
        A corresponding warning whenever this could be the case is issued.
        """

        # Shape derivative of Lagrangian w/o regularization and pull-backs
        self.shape_derivative = fenics.derivative(
            self.lagrangian_form,
            fenics.SpatialCoordinate(self.mesh),
            self.test_vector_field,
        )

        if self.use_scalar_tracking:
            for j in range(self.no_scalar_tracking_terms):
                self.shape_derivative += fenics.derivative(
                    self.scalar_weights[j]
                    * (
                        self.scalar_cost_functional_integrand_values[j]
                        - fenics.Constant(self.scalar_tracking_goals[j])
                    )
                    * self.scalar_cost_functional_integrands[j],
                    fenics.SpatialCoordinate(self.mesh),
                    self.test_vector_field,
                )

        if self.use_min_max_terms:
            for j in range(self.no_min_max_terms):
                if self.min_max_lower_bounds[j] is not None:
                    term_lower = self.min_max_lambda[j] + self.min_max_mu[j] * (
                        self.min_max_integrand_values[j] - self.min_max_lower_bounds[j]
                    )
                    self.shape_derivative += fenics.derivative(
                        _min(fenics.Constant(0.0), term_lower)
                        * self.min_max_integrands[j],
                        fenics.SpatialCoordinate(self.mesh),
                        self.test_vector_field,
                    )

                if self.min_max_upper_bounds[j] is not None:
                    term_upper = self.min_max_lambda[j] + self.min_max_mu[j] * (
                        self.min_max_integrand_values[j] - self.min_max_upper_bounds[j]
                    )
                    self.shape_derivative += fenics.derivative(
                        _max(fenics.Constant(0.0), term_upper)
                        * self.min_max_integrands[j],
                        fenics.SpatialCoordinate(self.mesh),
                        self.test_vector_field,
                    )

        # Add pull-backs
        if self.use_pull_back:
            self.state_adjoint_ids = [coeff.id() for coeff in self.states] + [
                coeff.id() for coeff in self.adjoints
            ]

            self.material_derivative_coeffs = []
            for coeff in self.lagrangian_form.coefficients():
                if coeff.id() in self.state_adjoint_ids:
                    pass
                else:
                    if not (coeff.ufl_element().family() == "Real"):
                        self.material_derivative_coeffs.append(coeff)

            if self.use_scalar_tracking:
                for j in range(self.no_scalar_tracking_terms):
                    for coeff in self.scalar_cost_functional_integrands[
                        j
                    ].coefficients():
                        if coeff.id() in self.state_adjoint_ids:
                            pass
                        else:
                            if not (coeff.ufl_element().family() == "Real"):
                                self.material_derivative_coeffs.append(coeff)

            if self.use_min_max_terms:
                for j in range(self.no_min_max_terms):
                    for coeff in self.min_max_integrands[j].coefficients():
                        if coeff.id() in self.state_adjoint_ids:
                            pass
                        else:
                            if not (coeff.ufl_element().family() == "Real"):
                                self.material_derivative_coeffs.append(coeff)

            if len(self.material_derivative_coeffs) > 0:
                warning(
                    "Shape derivative might be wrong, if differential operators act on variables other than states and adjoints. \n"
                    "You can check for correctness of the shape derivative with cashocs.verification.shape_gradient_test\n"
                )

            for coeff in self.material_derivative_coeffs:

                material_derivative = fenics.derivative(
                    self.lagrangian_form,
                    coeff,
                    fenics.dot(fenics.grad(coeff), self.test_vector_field),
                )
                if self.use_scalar_tracking:
                    for j in range(self.no_scalar_tracking_terms):
                        material_derivative += fenics.derivative(
                            self.scalar_weights[j]
                            * (
                                self.scalar_cost_functional_integrand_values[j]
                                - fenics.Constant(self.scalar_tracking_goals[j])
                            )
                            * self.scalar_cost_functional_integrands[j],
                            coeff,
                            fenics.dot(fenics.grad(coeff), self.test_vector_field),
                        )

                if self.use_min_max_terms:
                    for j in range(self.no_min_max_terms):
                        if self.min_max_lower_bounds[j] is not None:
                            term_lower = self.min_max_lambda[j] + self.min_max_mu[j] * (
                                self.min_max_integrand_values[j]
                                - self.min_max_lower_bounds[j]
                            )
                            material_derivative += fenics.derivative(
                                _min(fenics.Constant(0.0), term_lower)
                                * self.min_max_integrands[j],
                                coeff,
                                fenics.dot(fenics.grad(coeff), self.test_vector_field),
                            )

                        if self.min_max_upper_bounds[j] is not None:
                            term_upper = self.min_max_lambda[j] + self.min_max_mu[j] * (
                                self.min_max_integrand_values[j]
                                - self.min_max_upper_bounds[j]
                            )
                            material_derivative += fenics.derivative(
                                _max(fenics.Constant(0.0), term_upper)
                                * self.min_max_integrands[j],
                                coeff,
                                fenics.dot(fenics.grad(coeff), self.test_vector_field),
                            )

                material_derivative = expand_derivatives(material_derivative)

                self.shape_derivative += material_derivative

        # Add regularization
        self.shape_derivative += self.shape_regularization.compute_shape_derivative()

    def __compute_shape_gradient_forms(self) -> None:
        """Calculates the necessary left-hand-sides for the shape gradient problem.

        Returns
        -------
        None
        """

        shape_bdry_temp = self.config.get(
            "ShapeGradient", "shape_bdry_def", fallback="[]"
        )

        self.shape_bdry_def = json.loads(shape_bdry_temp)

        shape_bdry_temp = self.config.get(
            "ShapeGradient", "shape_bdry_fix", fallback="[]"
        )
        self.shape_bdry_fix = json.loads(shape_bdry_temp)

        shape_bdry_temp = self.config.get(
            "ShapeGradient", "shape_bdry_fix_x", fallback="[]"
        )
        self.shape_bdry_fix_x = json.loads(shape_bdry_temp)

        shape_bdry_temp = self.config.get(
            "ShapeGradient", "shape_bdry_fix_y", fallback="[]"
        )
        self.shape_bdry_fix_y = json.loads(shape_bdry_temp)

        shape_bdry_temp = self.config.get(
            "ShapeGradient", "shape_bdry_fix_z", fallback="[]"
        )
        self.shape_bdry_fix_z = json.loads(shape_bdry_temp)

        self.bcs_shape = create_dirichlet_bcs(
            self.deformation_space,
            fenics.Constant([0] * self.deformation_space.ufl_element().value_size()),
            self.boundaries,
            self.shape_bdry_fix,
        )
        self.bcs_shape += create_dirichlet_bcs(
            self.deformation_space.sub(0),
            fenics.Constant(0.0),
            self.boundaries,
            self.shape_bdry_fix_x,
        )
        self.bcs_shape += create_dirichlet_bcs(
            self.deformation_space.sub(1),
            fenics.Constant(0.0),
            self.boundaries,
            self.shape_bdry_fix_y,
        )
        if self.deformation_space.num_sub_spaces() == 3:
            self.bcs_shape += create_dirichlet_bcs(
                self.deformation_space.sub(2),
                fenics.Constant(0.0),
                self.boundaries,
                self.shape_bdry_fix_z,
            )

        self.CG1 = fenics.FunctionSpace(self.mesh, "CG", 1)
        self.DG0 = fenics.FunctionSpace(self.mesh, "DG", 0)

        self.mu_lame = fenics.Function(self.CG1)
        self.mu_lame.vector().vec().set(1.0)

        if self.shape_scalar_product is None:
            # Use the default linear elasticity approach

            self.lambda_lame = self.config.getfloat(
                "ShapeGradient", "lambda_lame", fallback=0.0
            )
            self.damping_factor = self.config.getfloat(
                "ShapeGradient", "damping_factor", fallback=0.0
            )

            if self.config.getboolean("ShapeGradient", "inhomogeneous", fallback=False):
                self.volumes = fenics.project(fenics.CellVolume(self.mesh), self.DG0)

                vol_max = self.volumes.vector().vec().max()[1]
                self.volumes.vector().vec().scale(1 / vol_max)

            else:
                self.volumes = fenics.Constant(1.0)

            def eps(u):
                """Computes the symmetrized gradient of a vector field ``u``.

                Parameters
                ----------
                u : fenics.Function
                    A vector field

                Returns
                -------
                ufl.core.expr.Expr
                    The symmetrized gradient of ``u``


                """
                return fenics.Constant(0.5) * (fenics.grad(u) + fenics.grad(u).T)

            trial = fenics.TrialFunction(self.deformation_space)
            test = fenics.TestFunction(self.deformation_space)

            self.riesz_scalar_product = (
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

    def __setup_mu_computation(self) -> None:

        if not self.use_distance_mu:
            self.mu_def = self.config.getfloat("ShapeGradient", "mu_def", fallback=1.0)
            self.mu_fix = self.config.getfloat("ShapeGradient", "mu_fix", fallback=1.0)

            if np.abs(self.mu_def - self.mu_fix) / self.mu_fix > 1e-2:

                self.inhomogeneous_mu = True

                self.options_mu = [
                    ["ksp_type", "cg"],
                    ["pc_type", "hypre"],
                    ["pc_hypre_type", "boomeramg"],
                    ["ksp_rtol", 1e-16],
                    ["ksp_atol", 1e-50],
                    ["ksp_max_it", 100],
                ]
                self.ksp_mu = PETSc.KSP().create()
                _setup_petsc_options([self.ksp_mu], [self.options_mu])

                phi = fenics.TrialFunction(self.CG1)
                psi = fenics.TestFunction(self.CG1)

                self.a_mu = fenics.inner(fenics.grad(phi), fenics.grad(psi)) * self.dx
                self.L_mu = fenics.Constant(0.0) * psi * self.dx

                self.bcs_mu = create_dirichlet_bcs(
                    self.CG1,
                    fenics.Constant(self.mu_fix),
                    self.boundaries,
                    self.shape_bdry_fix,
                )
                self.bcs_mu += create_dirichlet_bcs(
                    self.CG1,
                    fenics.Constant(self.mu_def),
                    self.boundaries,
                    self.shape_bdry_def,
                )

        else:
            self.mu_min = self.config.getfloat("ShapeGradient", "mu_min", fallback=1.0)
            self.mu_max = self.config.getfloat("ShapeGradient", "mu_max", fallback=1.0)

            if np.abs(self.mu_min - self.mu_max) / self.mu_min > 1e-2:
                self.dist_min = self.config.getfloat(
                    "ShapeGradient", "dist_min", fallback=1.0
                )
                self.dist_max = self.config.getfloat(
                    "ShapeGradient", "dist_max", fallback=1.0
                )
                if self.dist_min > self.dist_max:
                    raise IncompatibleConfigurationError(
                        "dist_max",
                        "ShapeGradient",
                        "dist_min",
                        "ShapeGradient",
                        "Reason: dist_max has to be larger than dist_min",
                    )

                self.bdry_idcs = json.loads(
                    self.config.get("ShapeGradient", "boundaries_dist", fallback="[]")
                )
                self.smooth_mu = self.config.getboolean(
                    "ShapeGradient", "smooth_mu", fallback=False
                )
                self.distance = fenics.Function(self.CG1)
                if not self.smooth_mu:
                    self.mu_expression = fenics.Expression(
                        "(dist <= dist_min) ? mu_min : "
                        + "(dist <= dist_max) ? mu_min + (dist - dist_min)/(dist_max - dist_min)*(mu_max - mu_min) : mu_max",
                        degree=1,
                        dist=self.distance,
                        dist_min=self.dist_min,
                        dist_max=self.dist_max,
                        mu_min=self.mu_min,
                        mu_max=self.mu_max,
                    )
                else:
                    self.mu_expression = fenics.Expression(
                        "(dist <= dist_min) ? mu_min :"
                        + "(dist <= dist_max) ? mu_min + (mu_max - mu_min)/(dist_max - dist_min)*(dist - dist_min) "
                        + "- (mu_max - mu_min)/pow(dist_max - dist_min, 2)*(dist - dist_min)*(dist - dist_max) "
                        + "- 2*(mu_max - mu_min)/pow(dist_max - dist_min, 3)*(dist - dist_min)*pow(dist - dist_max, 2)"
                        + " : mu_max",
                        degree=3,
                        dist=self.distance,
                        dist_min=self.dist_min,
                        dist_max=self.dist_max,
                        mu_min=self.mu_min,
                        mu_max=self.mu_max,
                    )

    def __compute_mu_elas(self) -> None:
        """Computes the second lame parameter mu_elas, based on `Schulz and
        Siebenborn, Computational Comparison of Surface Metrics for
        PDE Constrained Shape Optimization
        <https://doi.org/10.1515/cmam-2016-0009>`_

        Returns
        -------
        None
        """

        if self.shape_scalar_product is None:
            if not self.use_distance_mu:
                if self.inhomogeneous_mu:

                    _assemble_petsc_system(
                        self.a_mu,
                        self.L_mu,
                        self.bcs_mu,
                        A_tensor=self.A_mu,
                        b_tensor=self.b_mu,
                    )
                    x = _solve_linear_problem(
                        self.ksp_mu,
                        self.A_mu.mat(),
                        self.b_mu.vec(),
                        ksp_options=self.options_mu,
                    )

                    if self.config.getboolean(
                        "ShapeGradient", "use_sqrt_mu", fallback=False
                    ):
                        x.sqrtabs()

                    self.mu_lame.vector().vec().aypx(0.0, x)

                else:
                    self.mu_lame.vector().vec().set(self.mu_fix)

            else:
                self.distance.vector().vec().aypx(
                    0.0,
                    compute_boundary_distance(
                        self.mesh, self.boundaries, self.bdry_idcs
                    )
                    .vector()
                    .vec(),
                )
                self.mu_lame.vector().vec().aypx(
                    0.0, fenics.interpolate(self.mu_expression, self.CG1).vector().vec()
                )

            # for mpi compatibility
            self.mu_lame.vector().apply("")

    def _project_scalar_product(self) -> None:
        """Ensures, that only free dimensions can be deformed.

        Returns
        -------
        None

        """
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

        Returns
        -------
        None
        """

        self.__compute_mu_elas()
        if self.update_inhomogeneous:
            self.volumes.vector().vec().aypx(
                0.0,
                fenics.project(fenics.CellVolume(self.mesh), self.DG0).vector().vec(),
            )
            vol_max = self.volumes.vector().vec().max()[1]
            self.volumes.vector().vec().scale(1 / vol_max)

        self.assembler.assemble(self.fe_scalar_product_matrix)
        self.fe_scalar_product_matrix.ident_zeros()
        self.scalar_product_matrix = self.fe_scalar_product_matrix.mat()
        self._project_scalar_product()

    def scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between two deformation functions.

        Parameters
        ----------
        a : list[fenics.Function]
            The first argument.
        b : list[fenics.Function]
            The second argument.

        Returns
        -------
        float
            The value of the scalar product.
        """

        if (
            self.config.getboolean("ShapeGradient", "use_p_laplacian", fallback=False)
            and not self.uses_custom_scalar_product
        ):
            self.form = replace(
                self.F_p_laplace, {self.gradient[0]: a[0], self.test_vector_field: b[0]}
            )
            result = fenics.assemble(self.form)

        else:

            x = fenics.as_backend_type(a[0].vector()).vec()
            y = fenics.as_backend_type(b[0].vector()).vec()

            temp, _ = self.scalar_product_matrix.getVecs()
            self.scalar_product_matrix.mult(x, temp)
            result = temp.dot(y)

        return result

    def __compute_p_laplacian_forms(self) -> None:
        """Computes the weak forms for the p-Laplace equations, for computing the shape derivative

        Returns
        -------
        None

        """

        if self.config.getboolean("ShapeGradient", "use_p_laplacian", fallback=False):
            p = self.config.getint("ShapeGradient", "p_laplacian_power", fallback=2)
            delta = self.config.getfloat(
                "ShapeGradient", "damping_factor", fallback=0.0
            )
            eps = self.config.getfloat(
                "ShapeGradient", "p_laplacian_stabilization", fallback=0.0
            )
            kappa = pow(
                fenics.inner(
                    fenics.grad(self.gradient[0]), fenics.grad(self.gradient[0])
                ),
                (p - 2) / 2.0,
            )
            self.F_p_laplace = (
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
