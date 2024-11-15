# Copyright (C) 2020-2024 Fraunhofer ITWM and Sebastian Blauth
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

"""Shape gradient problem.

This class uses the linear elasticity equations to project the shape derivative to the
shape gradient with a Riesz projection.
"""

from __future__ import annotations

import configparser
import copy
from typing import List, TYPE_CHECKING

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _utils
from cashocs import log
from cashocs import nonlinear_solvers
from cashocs._pde_problems import pde_problem

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._pde_problems import adjoint_problem as ap
    from cashocs._pde_problems import state_problem as sp


class ShapeGradientProblem(pde_problem.PDEProblem):
    """Riesz problem for the computation of the shape gradient."""

    def __init__(
        self,
        db: database.Database,
        form_handler: _forms.ShapeFormHandler,
        state_problem: sp.StateProblem,
        adjoint_problem: ap.AdjointProblem,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The ShapeFormHandler object corresponding to the shape
                optimization problem.
            state_problem: The corresponding state problem.
            adjoint_problem: The corresponding adjoint problem.

        """
        super().__init__(db)

        self.form_handler = form_handler
        self.state_problem = state_problem
        self.adjoint_problem = adjoint_problem

        self.gradient_norm_squared = 1.0

        gradient_tol = self.config.getfloat("OptimizationRoutine", "gradient_tol")

        gradient_method = self.config.get("OptimizationRoutine", "gradient_method")

        if db.parameter_db.gradient_ksp_options is not None:
            self.ksp_options = db.parameter_db.gradient_ksp_options[0]
        elif gradient_method.casefold() == "direct":
            self.ksp_options = copy.deepcopy(_utils.linalg.direct_ksp_options)
        elif gradient_method.casefold() == "iterative":
            self.ksp_options = copy.deepcopy(_utils.linalg.iterative_ksp_options)
            self.ksp_options["ksp_rtol"] = gradient_tol

        if (
            self.config.getboolean("ShapeGradient", "use_p_laplacian")
            and self.form_handler.use_fixed_dimensions
        ):
            log.warning(
                "Incompatible config settings: "
                "use_p_laplacian and fixed_dimensions are incompatible. "
                "Falling back to use_p_laplacian=False."
            )

        if (
            self.config.getboolean("ShapeGradient", "use_p_laplacian")
            and not self.form_handler.uses_custom_scalar_product
            and not self.form_handler.use_fixed_dimensions
        ):
            self.p_laplace_projector = _PLaplaceProjector(
                self.db,
                self,
                self.db.function_db.gradient,
                self.form_handler.shape_derivative,
                self.form_handler.bcs_shape,
                self.config,
            )

    def solve(self) -> List[fenics.Function]:
        """Solves the Riesz projection problem to obtain the shape gradient.

        Returns:
            The function representing the shape gradient of the (reduced) cost
            functional.

        """
        self.state_problem.solve()
        self.adjoint_problem.solve()

        if not self.has_solution:
            log.begin("Computing the gradient deformation.", level=log.DEBUG)

            self.form_handler.shape_regularization.update_geometric_quantities()

            if (
                self.config.getboolean("ShapeGradient", "use_p_laplacian")
                and not self.form_handler.uses_custom_scalar_product
                and not self.form_handler.use_fixed_dimensions
            ):
                self.p_laplace_projector.solve()
                self.has_solution = True

            else:
                self.form_handler.assembler.assemble(
                    self.form_handler.fe_shape_derivative_vector
                )
                if self.form_handler.use_fixed_dimensions:
                    self.form_handler.fe_shape_derivative_vector.vec().setValues(
                        self.form_handler.fixed_indices,
                        np.array([0.0] * len(self.form_handler.fixed_indices)),
                    )
                    self.form_handler.fe_shape_derivative_vector.apply("")
                self.linear_solver.solve(
                    A=self.form_handler.scalar_product_matrix,
                    b=self.form_handler.fe_shape_derivative_vector.vec(),
                    fun=self.db.function_db.gradient[0],
                    ksp_options=self.ksp_options,
                )
                self.form_handler.apply_shape_bcs(self.db.function_db.gradient[0])

                self.has_solution = True

            self.db.callback.call_post()

            self.gradient_norm_squared = self.form_handler.scalar_product(
                self.db.function_db.gradient, self.db.function_db.gradient
            )
            log.end()

        return self.db.function_db.gradient


class _PLaplaceProjector:
    """A class for computing the gradient deformation with a p-Laplace projection."""

    def __init__(
        self,
        db: database.Database,
        gradient_problem: ShapeGradientProblem,
        gradient: List[fenics.Function],
        shape_derivative: ufl.Form,
        bcs_shape: List[fenics.DirichletBC],
        config: configparser.ConfigParser,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            gradient_problem: The shape gradient problem
            gradient: The fenics Function representing the gradient deformation
            shape_derivative: The ufl Form of the shape derivative
            bcs_shape: The boundary conditions for computing the gradient deformation
            config: The config for the optimization problem

        """
        self.db = db
        self.p_target = config.getint("ShapeGradient", "p_laplacian_power")
        delta = config.getfloat("ShapeGradient", "damping_factor")
        eps = config.getfloat("ShapeGradient", "p_laplacian_stabilization")
        self.p_list = np.arange(2, self.p_target + 1, 1)
        self.solution = gradient[0]
        self.shape_derivative = shape_derivative
        self.test_vector_field = gradient_problem.form_handler.test_vector_field
        self.bcs_shape = bcs_shape
        dx = self.db.geometry_db.dx
        self.mu_lame = gradient_problem.form_handler.mu_lame

        self.A_tensor = fenics.PETScMatrix()  # pylint: disable=invalid-name
        self.b_tensor = fenics.PETScVector()

        self.form_list = []
        for p in self.p_list:
            kappa = pow(
                fenics.inner(fenics.grad(self.solution), fenics.grad(self.solution)),
                (p - 2) / 2.0,
            )
            self.form_list.append(
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

            gradient_method = config.get("OptimizationRoutine", "gradient_method")
            gradient_tol = config.get("OptimizationRoutine", "gradient_tol")

            if db.parameter_db.gradient_ksp_options is not None:
                self.ksp_options = db.parameter_db.gradient_ksp_options[0]
            elif gradient_method.casefold() == "direct":
                self.ksp_options = copy.deepcopy(_utils.linalg.direct_ksp_options)
            elif gradient_method.casefold() == "iterative":
                self.ksp_options = copy.deepcopy(_utils.linalg.iterative_ksp_options)
                self.ksp_options["ksp_rtol"] = gradient_tol

    def solve(self) -> None:
        """Solves the p-Laplace problem for computing the shape gradient."""
        log.begin("Computing the gradient deformation with the p-Laplace approach.")

        self.solution.vector().vec().set(0.0)
        self.solution.vector().apply("")
        for nonlinear_form in self.form_list:
            petsc_options: _typing.KspOption = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "basic",
                "snes_ksp_ew": True,
            }
            petsc_options.update(self.ksp_options)

            nonlinear_solvers.snes_solve(
                nonlinear_form,
                self.solution,
                self.bcs_shape,
                petsc_options=petsc_options,
                shift=self.shape_derivative,
                A_tensor=self.A_tensor,
                b_tensor=self.b_tensor,
            )

        log.end()
