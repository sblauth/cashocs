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

"""Shape gradient problem.

This class uses the linear elasticity equations to project the shape derivative to the
shape gradient with a Riesz projection.
"""

from __future__ import annotations

import configparser
import copy
from typing import TYPE_CHECKING

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs import log
from cashocs import nonlinear_solvers
from cashocs._pde_problems import pde_problem
from cashocs.geometry import measure

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

    def solve(self) -> list[fenics.Function]:
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
                self.db.function_db.gradient[0].vector().vec().set(0.0)
                self.db.function_db.gradient[0].vector().apply("")
                self.linear_solver.solve(
                    self.db.function_db.gradient[0],
                    A=self.form_handler.scalar_product_matrix,
                    b=self.form_handler.fe_shape_derivative_vector.vec(),
                    ksp_options=self.ksp_options,
                )
                self.form_handler.apply_shape_bcs(self.db.function_db.gradient[0])

                self.has_solution = True

            self.reextend_gradient_from_boundaries()
            self.db.callback.call_post()

            self.gradient_norm_squared = self.form_handler.scalar_product(
                self.db.function_db.gradient, self.db.function_db.gradient
            )
            log.end()

        return self.db.function_db.gradient

    def reextend_gradient_from_boundaries(self) -> None:
        """Re-extends the gradient to the volume from its boundary values.

        Note:
            This method modifies the gradient deformation and the original one is
            over-written.

        """
        if self.config.getboolean("ShapeGradient", "reextend_from_boundary"):
            log.debug("Re-extending the gradient deformation from the boundary.")

            if self.config.get("ShapeGradient", "reextension_mode") == "normal":
                normal_deformation = self._compute_normal_deformation()
                self.db.function_db.gradient[0].vector().vec().aypx(
                    0.0, normal_deformation.vector().vec()
                )
                self.db.function_db.gradient[0].vector().apply("")

            self.form_handler.assembler_extension.assemble(
                self.form_handler.fe_reextension_vector
            )
            if self.form_handler.use_fixed_dimensions:
                self.form_handler.fe_reextension_vector.vec().setValues(
                    self.form_handler.fixed_indices,
                    np.array([0.0] * len(self.form_handler.fixed_indices)),
                )
                self.form_handler.fe_reextension_vector.apply("")

            reextended_gradient = fenics.Function(self.db.function_db.control_spaces[0])
            self.form_handler.apply_reextension_bcs(  # Effect of DirichletBCs on KSP
                reextended_gradient
            )
            self.linear_solver.solve(
                reextended_gradient,
                A=self.form_handler.reextension_matrix,
                b=self.form_handler.fe_reextension_vector.vec(),
                ksp_options=self.ksp_options,
            )
            self.form_handler.apply_reextension_bcs(reextended_gradient)

            self.db.function_db.gradient[0].vector().vec().aypx(
                0.0, reextended_gradient.vector().vec()
            )
            self.db.function_db.gradient[0].vector().apply("")

    def _compute_normal_deformation(self) -> fenics.Function:
        """Computes the projection of the gradient deformation on the normal vector.

        Returns:
            The normal component gradient deformation in normal direction.

        """
        mesh = self.db.geometry_db.mesh

        if fenics.parameters["ghost_mode"] == "none" and mesh.mpi_comm().size > 1:
            raise _exceptions.CashocsException(
                "Cannot re-extend the gradient deformation based on the normal "
                + "deformation. "
                + "Try using fenics.parameters['ghost_mode'] = 'shared_facet' "
                + "or 'shared_vertex' before importing your mesh."
            )

        physical_groups = None
        if hasattr(mesh, "physical_groups"):
            physical_groups = mesh.physical_groups

        ds = measure.NamedMeasure(
            "ds",
            domain=mesh,
            subdomain_data=self.form_handler.boundaries,
            physical_groups=physical_groups,
        )
        dS = measure.NamedMeasure(  # pylint: disable=invalid-name
            "dS",
            domain=mesh,
            subdomain_data=self.form_handler.boundaries,
            physical_groups=physical_groups,
        )

        n = fenics.FacetNormal(mesh)
        all_boundaries = (
            self.form_handler.shape_bdry_def
            + self.form_handler.shape_bdry_fix
            + self.form_handler.shape_bdry_fix_x
            + self.form_handler.shape_bdry_fix_y
            + self.form_handler.shape_bdry_fix_z
        )
        space = fenics.FunctionSpace(mesh, "CG", 1)
        normal_deformation = fenics.Function(space)
        v = fenics.TestFunction(space)

        normal_integrand = (
            normal_deformation * v - ufl.dot(self.db.function_db.gradient[0], n) * v
        )
        normal_form = normal_integrand * ds(all_boundaries) + normal_integrand(
            "+"
        ) * dS(all_boundaries)

        bcs: list[fenics.DirichletBC] = []

        ksp_options: _typing.KspOption = {
            "ksp_type": "cg",
            "ksp_rtol": 1e-8,
            "ksp_atol": 1e-30,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
        }
        nonlinear_solvers.linear_solve(
            normal_form, normal_deformation, bcs, ksp_options=ksp_options
        )

        surface_deformation = fenics.Function(self.db.function_db.control_spaces[0])
        v = fenics.TestFunction(self.db.function_db.control_spaces[0])
        surface_integrand = ufl.dot(
            surface_deformation, v
        ) - normal_deformation * ufl.dot(n, v)

        surface_form = surface_integrand * ds(all_boundaries) + surface_integrand(
            "+"
        ) * dS(all_boundaries)
        nonlinear_solvers.linear_solve(
            surface_form, surface_deformation, bcs, ksp_options=ksp_options
        )

        return surface_deformation


class _PLaplaceProjector:
    """A class for computing the gradient deformation with a p-Laplace projection."""

    def __init__(
        self,
        db: database.Database,
        gradient_problem: ShapeGradientProblem,
        gradient: list[fenics.Function],
        shape_derivative: ufl.Form,
        bcs_shape: list[fenics.DirichletBC],
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

        self.A_tensor = fenics.PETScMatrix(  # pylint: disable=invalid-name
            self.db.geometry_db.mpi_comm
        )
        self.b_tensor = fenics.PETScVector(self.db.geometry_db.mpi_comm)

        self.form_list = []
        for p in self.p_list:
            kappa = pow(
                ufl.inner(ufl.grad(self.solution), ufl.grad(self.solution)),
                (p - 2) / 2.0,
            )
            self.form_list.append(
                ufl.inner(
                    self.mu_lame
                    * (fenics.Constant(eps) + kappa)
                    * ufl.grad(self.solution),
                    ufl.grad(self.test_vector_field),
                )
                * dx
                + fenics.Constant(delta)
                * ufl.dot(self.solution, self.test_vector_field)
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
        log.begin(
            "Computing the gradient deformation with the p-Laplace approach.",
            level=log.DEBUG,
        )

        self.solution.vector().vec().set(0.0)
        self.solution.vector().apply("")
        for nonlinear_form in self.form_list:
            petsc_options: _typing.KspOption = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "basic",
                "snes_ksp_ew": None,
                "snes_ksp_ew_rtolmax": 1e-1,
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
