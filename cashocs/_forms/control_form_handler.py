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

"""Management of weak forms for optimal control problems."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
from petsc4py import PETSc
import ufl
import ufl.algorithms

from cashocs import _exceptions
from cashocs import _utils
from cashocs._forms import form_handler
from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import optimal_control


class ControlFormHandler(form_handler.FormHandler):
    """Class for UFL form manipulation for optimal control problems.

    This is used to symbolically derive the corresponding weak forms of the
    adjoint and gradient equations (via UFL) , that are later used in the
    solvers for the equations later on. These are needed as subroutines for
     the optimization (solution) algorithms.
    """

    def __init__(
        self,
        optimization_problem: optimal_control.OptimalControlProblem,
        db: database.Database,
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimal control problem
            db: The database of the problem.

        """
        super().__init__(optimization_problem, db)

        self.hessian_form_handler = HessianFormHandler(self.db)
        self.riesz_scalar_products = optimization_problem.riesz_scalar_products
        self.control_bcs_list = optimization_problem.control_bcs_list

        self.gradient_forms_rhs: List[ufl.Form] = []
        self._compute_gradient_equations()

        self.modified_scalar_product: Optional[List[ufl.Form]] = None
        self.assemblers: List[fenics.SystemAssembler] = []
        self.riesz_projection_matrices: List[PETSc.Mat] = []
        self.setup_assemblers(
            self.riesz_scalar_products, self.gradient_forms_rhs, self.control_bcs_list
        )

    def setup_assemblers(
        self,
        scalar_product_forms: List[ufl.Form],
        derivatives: List[ufl.Form],
        bcs: Union[List[List[fenics.DirichletBC]], List[None]],
    ) -> None:
        """Sets up the assemblers and matrices for the projection of the gradient.

        Args:
            scalar_product_forms: The weak form of the scalar product.
            derivatives: The weak form of the derivative.
            bcs: The boundary conditions for the projection.

        """
        modified_scalar_product_forms = _utils.bilinear_boundary_form_modification(
            scalar_product_forms
        )
        self.modified_scalar_product = _utils.bilinear_boundary_form_modification(
            scalar_product_forms
        )
        try:
            self.assemblers.clear()
            for i in range(len(bcs)):
                assembler = fenics.SystemAssembler(
                    modified_scalar_product_forms[i],
                    derivatives[i],
                    bcs[i],
                )
                assembler.keep_diagonal = True
                self.assemblers.append(assembler)
        except (AssertionError, ValueError):
            self.assemblers.clear()
            for i in range(len(self.riesz_scalar_products)):
                estimated_degree = np.maximum(
                    ufl.algorithms.estimate_total_polynomial_degree(
                        self.riesz_scalar_products[i]
                    ),
                    ufl.algorithms.estimate_total_polynomial_degree(
                        self.gradient_forms_rhs[i]
                    ),
                )
                assembler = fenics.SystemAssembler(
                    modified_scalar_product_forms[i],
                    derivatives[i],
                    bcs[i],
                    form_compiler_parameters={"quadrature_degree": estimated_degree},
                )
                assembler.keep_diagonal = True
                self.assemblers.append(assembler)

        fenics_scalar_product_matrices = []

        for i in range(len(self.assemblers)):
            fenics_matrix = fenics.PETScMatrix()
            fenics_scalar_product_matrices.append(fenics_matrix)

            self.assemblers[i].assemble(fenics_matrix)
            fenics_scalar_product_matrices[i].ident_zeros()

            self.riesz_projection_matrices.append(fenics_matrix.mat())

        # Test for symmetry of the scalar products
        for matrix in self.riesz_projection_matrices:
            if not matrix.isSymmetric():
                if not matrix.isSymmetric(1e-15):
                    if (
                        not (matrix - matrix.copy().transpose()).norm() / matrix.norm()
                        < 1e-15
                    ):
                        raise _exceptions.InputError(
                            "cashocs._forms.ControlFormHandler",
                            "riesz_scalar_products",
                            "Supplied scalar product form is not symmetric.",
                        )

    def scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between control type functions a and b.

        Args:
            a: The first argument.
            b: The second argument.

        Returns:
            The value of the scalar product.

        """
        result = 0.0

        for i in range(len(a)):
            x = fenics.as_backend_type(a[i].vector()).vec()
            y = fenics.as_backend_type(b[i].vector()).vec()

            temp, _ = self.riesz_projection_matrices[i].getVecs()
            self.riesz_projection_matrices[i].mult(x, temp)
            result += temp.dot(y)

        return result

    def _compute_gradient_equations(self) -> None:
        """Calculates the variational form of the gradient equation."""
        test_functions_control = [
            fenics.TestFunction(c.function_space())
            for c in self.db.function_db.controls
        ]
        self.gradient_forms_rhs = [
            self.lagrangian.derivative(
                self.db.function_db.controls[i], test_functions_control[i]
            )
            for i in range(len(self.db.function_db.controls))
        ]


class HessianFormHandler:
    """Form handler for second order forms and hessians."""

    def __init__(self, db: database.Database):
        """Initializes the form handler for the second derivatives.

        Args:
            db: The database of the problem.

        """
        self.db = db

        self.config = self.db.config

        self.w_1: List[ufl.Form] = []
        self.w_2: List[ufl.Form] = []
        self.w_3: List[ufl.Form] = []
        self.hessian_rhs: List[ufl.Form] = []
        self.test_directions = _utils.create_function_list(
            [c.function_space() for c in self.db.function_db.controls]
        )
        self.test_functions_control = [
            fenics.TestFunction(c.function_space())
            for c in self.db.function_db.controls
        ]

        self.sensitivity_eqs_temp: List[ufl.Form] = []
        self.sensitivity_eqs_lhs: List[ufl.Form] = []
        self.sensitivity_eqs_picard: List[ufl.Form] = []
        self.sensitivity_eqs_rhs: List[ufl.Form] = []
        self.lagrangian_y: List[ufl.Form] = []
        self.lagrangian_u: List[ufl.Form] = []
        self.lagrangian_yy: List[List[ufl.Form]] = []
        self.lagrangian_yu: List[List[ufl.Form]] = []
        self.lagrangian_uy: List[List[ufl.Form]] = []
        self.lagrangian_uu: List[List[ufl.Form]] = []
        self.adjoint_sensitivity_eqs_lhs: List[ufl.Form] = []
        self.adjoint_sensitivity_eqs_picard: List[ufl.Form] = []
        self.adjoint_sensitivity_eqs_rhs: List[ufl.Form] = []

        opt_algo = _utils.optimization_algorithm_configuration(self.config)
        if opt_algo.casefold() == "newton":
            self.compute_newton_forms()

    def _compute_sensitivity_equations(self) -> None:
        """Calculates the forms for the (forward) sensitivity equations."""
        # Use replace -> derivative to speed up the computations
        self.sensitivity_eqs_temp = [
            ufl.replace(
                self.db.form_db.state_forms[i],
                {
                    self.db.function_db.adjoints[
                        i
                    ]: self.db.function_db.test_functions_state[i]
                },
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        self.sensitivity_eqs_lhs = [
            fenics.derivative(
                self.sensitivity_eqs_temp[i],
                self.db.function_db.states[i],
                self.db.function_db.trial_functions_state[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]
        if self.config.getboolean("StateSystem", "picard_iteration"):
            self.sensitivity_eqs_picard = [
                fenics.derivative(
                    self.sensitivity_eqs_temp[i],
                    self.db.function_db.states[i],
                    self.db.function_db.states_prime[i],
                )
                for i in range(self.db.parameter_db.state_dim)
            ]

        # Need to distinguish cases due to empty sum in case state_dim = 1
        if self.db.parameter_db.state_dim > 1:
            # pylint: disable=invalid-unary-operand-type
            self.sensitivity_eqs_rhs = [
                -_utils.summation(
                    [
                        fenics.derivative(
                            self.sensitivity_eqs_temp[i],
                            self.db.function_db.states[j],
                            self.db.function_db.states_prime[j],
                        )
                        for j in range(self.db.parameter_db.state_dim)
                        if j != i
                    ]
                )
                - _utils.summation(
                    [
                        fenics.derivative(
                            self.sensitivity_eqs_temp[i],
                            self.db.function_db.controls[j],
                            self.test_directions[j],
                        )
                        for j in range(len(self.db.function_db.controls))
                    ]
                )
                for i in range(self.db.parameter_db.state_dim)
            ]
        else:
            self.sensitivity_eqs_rhs = [
                # pylint: disable=invalid-unary-operand-type
                -_utils.summation(
                    [
                        fenics.derivative(
                            self.sensitivity_eqs_temp[i],
                            self.db.function_db.controls[j],
                            self.test_directions[j],
                        )
                        for j in range(len(self.db.function_db.controls))
                    ]
                )
                for i in range(self.db.parameter_db.state_dim)
            ]

        # Add the right-hand-side for the picard iteration
        if self.config.getboolean("StateSystem", "picard_iteration"):
            for i in range(self.db.parameter_db.state_dim):
                self.sensitivity_eqs_picard[i] -= self.sensitivity_eqs_rhs[i]

    def _compute_first_order_lagrangian_derivatives(self) -> None:
        """Computes the derivative of the Lagrangian w.r.t. the state and control."""
        self.lagrangian_y = [
            self.db.form_db.lagrangian.derivative(
                self.db.function_db.states[i],
                self.db.function_db.test_functions_state[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]
        self.lagrangian_u = [
            self.db.form_db.lagrangian.derivative(
                self.db.function_db.controls[i], self.test_functions_control[i]
            )
            for i in range(len(self.db.function_db.controls))
        ]

    def _compute_second_order_lagrangian_derivatives(self) -> None:
        """Compute the second order derivatives of the Lagrangian w.r.t. y and u."""
        self.lagrangian_yy = [
            [
                fenics.derivative(
                    self.lagrangian_y[i],
                    self.db.function_db.states[j],
                    self.db.function_db.states_prime[j],
                )
                for j in range(self.db.parameter_db.state_dim)
            ]
            for i in range(self.db.parameter_db.state_dim)
        ]
        self.lagrangian_yu = [
            [
                fenics.derivative(
                    self.lagrangian_u[i],
                    self.db.function_db.states[j],
                    self.db.function_db.states_prime[j],
                )
                for j in range(self.db.parameter_db.state_dim)
            ]
            for i in range(len(self.lagrangian_u))
        ]
        self.lagrangian_uy = [
            [
                fenics.derivative(
                    self.lagrangian_y[i],
                    self.db.function_db.controls[j],
                    self.test_directions[j],
                )
                for j in range(len(self.db.function_db.controls))
            ]
            for i in range(self.db.parameter_db.state_dim)
        ]
        self.lagrangian_uu = [
            [
                fenics.derivative(
                    self.lagrangian_u[i],
                    self.db.function_db.controls[j],
                    self.test_directions[j],
                )
                for j in range(len(self.db.function_db.controls))
            ]
            for i in range(len(self.lagrangian_u))
        ]

    def _compute_adjoint_sensitivity_equations(self) -> None:
        """Computes the adjoint sensitivity equations for the Newton method."""
        # Use replace -> derivative for faster computations
        adjoint_sensitivity_eqs_diag_temp = [
            ufl.replace(
                self.db.form_db.state_forms[i],
                {
                    self.db.function_db.adjoints[
                        i
                    ]: self.db.function_db.trial_functions_adjoint[i]
                },
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        mapping_dict = {
            self.db.function_db.adjoints[j]: self.db.function_db.adjoints_prime[j]
            for j in range(self.db.parameter_db.state_dim)
        }
        adjoint_sensitivity_eqs_all_temp = [
            ufl.replace(self.db.form_db.state_forms[i], mapping_dict)
            for i in range(self.db.parameter_db.state_dim)
        ]

        self.adjoint_sensitivity_eqs_lhs = [
            fenics.derivative(
                adjoint_sensitivity_eqs_diag_temp[i],
                self.db.function_db.states[i],
                self.db.function_db.test_functions_adjoint[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]
        if self.config.getboolean("StateSystem", "picard_iteration"):
            self.adjoint_sensitivity_eqs_picard = [
                fenics.derivative(
                    adjoint_sensitivity_eqs_all_temp[i],
                    self.db.function_db.states[i],
                    self.db.function_db.test_functions_adjoint[i],
                )
                for i in range(self.db.parameter_db.state_dim)
            ]

        # Need cases distinction due to empty sum for state_dim == 1
        if self.db.parameter_db.state_dim > 1:
            for i in range(self.db.parameter_db.state_dim):
                self.w_1[i] -= _utils.summation(
                    [
                        fenics.derivative(
                            adjoint_sensitivity_eqs_all_temp[j],
                            self.db.function_db.states[i],
                            self.db.function_db.test_functions_adjoint[i],
                        )
                        for j in range(self.db.parameter_db.state_dim)
                        if j != i
                    ]
                )
        else:
            pass

        # Add right-hand-side for picard iteration
        if self.config.getboolean("StateSystem", "picard_iteration"):
            for i in range(self.db.parameter_db.state_dim):
                self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]

        self.adjoint_sensitivity_eqs_rhs = [
            _utils.summation(
                [
                    fenics.derivative(
                        adjoint_sensitivity_eqs_all_temp[j],
                        self.db.function_db.controls[i],
                        self.test_functions_control[i],
                    )
                    for j in range(self.db.parameter_db.state_dim)
                ]
            )
            for i in range(len(self.db.function_db.controls))
        ]

    def compute_newton_forms(self) -> None:
        """Calculates the needed forms for the truncated Newton method."""
        if any(
            isinstance(
                functional,
                (
                    cost_functional.ScalarTrackingFunctional,
                    cost_functional.MinMaxFunctional,
                ),
            )
            for functional in self.db.form_db.cost_functional_list
        ):
            raise _exceptions.InputError(
                "cashocs._forms.ShapeFormHandler",
                "_compute_newton_forms",
                (
                    "Newton's method is not available with scalar tracking or"
                    " min_max terms."
                ),
            )

        self._compute_sensitivity_equations()
        self._compute_first_order_lagrangian_derivatives()
        self._compute_second_order_lagrangian_derivatives()

        self.w_1 = [
            _utils.summation(self.lagrangian_yy[i])
            + _utils.summation(self.lagrangian_uy[i])
            for i in range(self.db.parameter_db.state_dim)
        ]
        self.w_2 = [
            _utils.summation(self.lagrangian_yu[i])
            + _utils.summation(self.lagrangian_uu[i])
            for i in range(len(self.lagrangian_uu))
        ]

        self._compute_adjoint_sensitivity_equations()

        for i in range(len(self.w_2)):
            w3_i = -self.adjoint_sensitivity_eqs_rhs[i]
            self.w_3.append(w3_i)
            self.hessian_rhs.append(self.w_2[i] + self.w_3[i])
