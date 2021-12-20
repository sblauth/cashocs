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

"""Abstract implementation of a shape gradient problem.

This class uses the linear elasticity equations to project the
shape derivative to the shape gradient with a Riesz projection.
"""

from __future__ import annotations

import configparser
from typing import TYPE_CHECKING, List

import fenics
import numpy as np
import ufl
from petsc4py import PETSc

from .._interfaces.pde_problem import PDEProblem
from ..nonlinear_solvers import newton_solve
from ..utils import _setup_petsc_options, _solve_linear_problem


if TYPE_CHECKING:
    from .state_problem import StateProblem
    from .adjoint_problem import AdjointProblem
    from .._forms import ShapeFormHandler


class ShapeGradientProblem(PDEProblem):
    """Riesz problem for the computation of the shape gradient."""

    def __init__(
        self,
        form_handler: ShapeFormHandler,
        state_problem: StateProblem,
        adjoint_problem: AdjointProblem,
    ) -> None:
        """Initialize the ShapeGradientProblem.

        Parameters
        ----------
        form_handler : ShapeFormHandler
            The ShapeFormHandler object corresponding to the shape optimization problem.
        state_problem : StateProblem
            The corresponding state problem.
        adjoint_problem : AdjointProblem
            The corresponding adjoint problem.
        """

        super().__init__(form_handler)
        self.state_problem = state_problem
        self.adjoint_problem = adjoint_problem

        self.gradient = self.form_handler.gradient
        self.gradient_norm_squared = 1.0

        gradient_tol = self.config.getfloat(
            "OptimizationRoutine", "gradient_tol", fallback=1e-9
        )

        # Generate the Krylov solver for the shape gradient problem
        self.ksp = PETSc.KSP().create()

        gradient_method = self.config.get(
            "OptimizationRoutine", "gradient_method", fallback="direct"
        )

        if gradient_method == "direct":
            self.ksp_options = [
                ["ksp_type", "preonly"],
                ["pc_type", "lu"],
                ["pc_factor_mat_solver_type", "mumps"],
                ["mat_mumps_icntl_24", 1],
            ]
        elif gradient_method == "iterative":
            self.ksp_options = [
                ["ksp_type", "cg"],
                ["pc_type", "hypre"],
                ["pc_hypre_type", "boomeramg"],
                ["pc_hypre_boomeramg_strong_threshold", 0.7],
                ["ksp_rtol", gradient_tol],
                ["ksp_atol", 1e-50],
                ["ksp_max_it", 250],
            ]

        _setup_petsc_options([self.ksp], [self.ksp_options])

        if (
            self.config.getboolean("ShapeGradient", "use_p_laplacian", fallback=False)
            and not self.form_handler.uses_custom_scalar_product
        ):
            self.p_laplace_projector = _PLaplacProjector(
                self,
                self.gradient,
                self.form_handler.shape_derivative,
                self.form_handler.bcs_shape,
                self.config,
            )

    def solve(self) -> fenics.Function:
        """Solves the Riesz projection problem to obtain the shape gradient of the cost functional.

        Returns
        -------
        fenics.Function
            The function representing the shape gradient of the (reduced) cost functional.
        """

        self.state_problem.solve()
        self.adjoint_problem.solve()

        if not self.has_solution:

            self.form_handler.regularization.update_geometric_quantities()

            if (
                self.config.getboolean(
                    "ShapeGradient", "use_p_laplacian", fallback=False
                )
                and not self.form_handler.uses_custom_scalar_product
            ):
                self.p_laplace_projector.solve()
                self.has_solution = True

                self.gradient_norm_squared = self.form_handler.scalar_product(
                    self.gradient, self.gradient
                )

            else:
                self.form_handler.assembler.assemble(
                    self.form_handler.fe_shape_derivative_vector
                )
                if self.form_handler.use_fixed_dimensions:
                    self.form_handler.fe_shape_derivative_vector[
                        self.form_handler.fixed_indices
                    ] = 0.0
                _solve_linear_problem(
                    self.ksp,
                    self.form_handler.scalar_product_matrix,
                    self.form_handler.fe_shape_derivative_vector.vec(),
                    self.gradient.vector().vec(),
                    self.ksp_options,
                )
                self.gradient.vector().apply("")

                self.has_solution = True

                self.gradient_norm_squared = self.form_handler.scalar_product(
                    self.gradient, self.gradient
                )

            self.form_handler._post_hook()

        return self.gradient


class _PLaplacProjector:
    """A class for computing the gradient deformation with a p-Laplace projection"""

    def __init__(
        self,
        shape_gradient_problem: ShapeGradientProblem,
        gradient: fenics.Function,
        shape_derivative: ufl.Form,
        bcs_shape: List[fenics.DirichletBC],
        config: configparser.ConfigParser,
    ) -> None:
        """

        Parameters
        ----------
        shape_gradient_problem : ShapeGradientProblem
            The shape gradient problem
        gradient : fenics.Function
            The fenics Function representing the gradient deformation
        shape_derivative : ufl.Form
            The ufl Form of the shape derivative
        bcs_shape : list[fenics.DirichletBC]
            The boundary conditions for computing the gradient deformation
        config : configparser.ConfigParser
            The config for the optimization problem
        """

        self.p_target = config.getint("ShapeGradient", "p_laplacian_power", fallback=2)
        delta = config.getfloat("ShapeGradient", "damping_factor", fallback=0.0)
        eps = config.getfloat(
            "ShapeGradient", "p_laplacian_stabilization", fallback=0.0
        )
        self.p_list = np.arange(2, self.p_target + 1, 1)
        self.solution = gradient
        self.shape_derivative = shape_derivative
        self.test_vector_field = shape_gradient_problem.form_handler.test_vector_field
        self.bcs_shape = bcs_shape
        dx = shape_gradient_problem.form_handler.dx
        self.mu_lame = shape_gradient_problem.form_handler.mu_lame

        self.A_tensor = fenics.PETScMatrix()
        self.b_tensor = fenics.PETScVector()

        self.F_list = []
        for p in self.p_list:
            kappa = pow(
                fenics.inner(fenics.grad(self.solution), fenics.grad(self.solution)),
                (p - 2) / 2.0,
            )
            self.F_list.append(
                fenics.inner(
                    self.mu_lame
                    * (fenics.Constant(eps) + kappa)
                    * fenics.grad(self.solution),
                    fenics.grad(self.test_vector_field),
                )
                * dx
                + fenics.Constant(delta)
                * fenics.dot(self.solution, self.test_vector_field)
                * dx
            )

            gradient_method = config.get(
                "OptimizationRoutine", "gradient_method", fallback="direct"
            )

            if gradient_method == "direct":
                self.ksp_options = [
                    ["ksp_type", "preonly"],
                    ["pc_type", "lu"],
                    ["pc_factor_mat_solver_type", "mumps"],
                    ["mat_mumps_icntl_24", 1],
                ]
            elif gradient_method == "iterative":
                self.ksp_options = [
                    ["ksp_type", "cg"],
                    ["pc_type", "hypre"],
                    ["pc_hypre_type", "boomeramg"],
                    ["ksp_rtol", 1e-16],
                    ["ksp_atol", 1e-50],
                    ["ksp_max_it", 100],
                ]

    def solve(self) -> None:
        """Solves the p-Laplace problem for computing the shape gradient

        Returns
        -------
        None
        """

        self.solution.vector().vec().set(0.0)
        for F in self.F_list:
            newton_solve(
                F,
                self.solution,
                self.bcs_shape,
                shift=self.shape_derivative,
                damped=False,
                inexact=True,
                verbose=False,
                ksp_options=self.ksp_options,
                A_tensor=self.A_tensor,
                b_tensor=self.b_tensor,
            )
