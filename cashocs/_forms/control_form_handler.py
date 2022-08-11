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

"""Module for managing UFL forms for optimal control problems."""

from __future__ import annotations

from typing import List, TYPE_CHECKING, Union

import fenics
import numpy as np
import ufl
import ufl.algorithms

from cashocs import _exceptions
from cashocs import _utils
from cashocs._forms import form_handler
from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    from cashocs._optimization import optimal_control


class ControlFormHandler(form_handler.FormHandler):
    """Class for UFL form manipulation for optimal control problems.

    This is used to symbolically derive the corresponding weak forms of the
    adjoint and gradient equations (via UFL) , that are later used in the
    solvers for the equations later on. These are needed as subroutines for
     the optimization (solution) algorithms.
    """

    idx_active: List
    idx_active_lower: List
    idx_active_upper: List
    idx_inactive: List
    adjoint_sensitivity_eqs_picard: List[ufl.Form]
    adjoint_sensitivity_eqs_rhs: List[ufl.Form]
    w_1: List[ufl.Form]
    w_2: List[ufl.Form]
    w_3: List[ufl.Form]
    hessian_rhs: List[ufl.Form]
    states_prime: List[fenics.Function]
    adjoints_prime: List[fenics.Function]
    test_directions: List[fenics.Function]
    test_functions_control: List[fenics.Function]
    temp: List[fenics.Function]

    def __init__(
        self, optimization_problem: optimal_control.OptimalControlProblem
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimal control problem

        """
        super().__init__(optimization_problem)

        # Initialize the attributes from the arguments
        self.controls = optimization_problem.controls
        self.riesz_scalar_products = optimization_problem.riesz_scalar_products
        self.control_bcs_list = optimization_problem.control_bcs_list
        self.control_constraints = optimization_problem.control_constraints
        self.require_control_constraints = (
            optimization_problem.require_control_constraints
        )

        self.control_dim = len(self.controls)
        self.control_spaces = [x.function_space() for x in self.controls]

        self.gradient = _utils.create_function_list(self.control_spaces)

        self._init_helpers()
        self._compute_gradient_equations()

        if self.opt_algo.casefold() == "newton":
            self.compute_newton_forms()

        self.setup_assemblers(
            self.riesz_scalar_products, self.gradient_forms_rhs, self.control_bcs_list
        )

    def _init_helpers(self) -> None:
        """Initializes the helper functions needed for the form handler."""
        self.states_prime = _utils.create_function_list(self.state_spaces)
        self.adjoints_prime = _utils.create_function_list(self.adjoint_spaces)
        self.test_directions = _utils.create_function_list(self.control_spaces)
        self.temp = _utils.create_function_list(self.control_spaces)
        self.test_functions_control = [
            fenics.TestFunction(function_space)
            for function_space in self.control_spaces
        ]

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
            self.assemblers = []
            for i in range(self.control_dim):
                assembler = fenics.SystemAssembler(
                    modified_scalar_product_forms[i],
                    derivatives[i],
                    bcs[i],
                )
                assembler.keep_diagonal = True
                self.assemblers.append(assembler)
        except (AssertionError, ValueError):
            self.assemblers = []
            for i in range(self.control_dim):
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
        self.riesz_projection_matrices = []

        for i in range(self.control_dim):
            fenics_matrix = fenics.PETScMatrix()
            fenics_scalar_product_matrices.append(fenics_matrix)

            self.assemblers[i].assemble(fenics_matrix)
            fenics_scalar_product_matrices[i].ident_zeros()

            self.riesz_projection_matrices.append(fenics_matrix.mat())

        # Test for symmetry of the scalar products
        for i in range(self.control_dim):
            if not self.riesz_projection_matrices[i].isSymmetric():
                if not self.riesz_projection_matrices[i].isSymmetric(1e-15):
                    if (
                        not (
                            self.riesz_projection_matrices[i]
                            - self.riesz_projection_matrices[i].copy().transpose()
                        ).norm()
                        / self.riesz_projection_matrices[i].norm()
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

        for i in range(self.control_dim):
            x = fenics.as_backend_type(a[i].vector()).vec()
            y = fenics.as_backend_type(b[i].vector()).vec()

            temp, _ = self.riesz_projection_matrices[i].getVecs()
            self.riesz_projection_matrices[i].mult(x, temp)
            result += temp.dot(y)

        return result

    def compute_active_sets(self) -> None:
        """Computes the indices corresponding to active and inactive sets."""
        self.idx_active_lower = []
        self.idx_active_upper = []
        self.idx_active = []
        self.idx_inactive = []

        for j in range(self.control_dim):

            if self.require_control_constraints[j]:
                self.idx_active_lower.append(
                    np.nonzero(
                        (
                            self.controls[j].vector()[:]
                            <= self.control_constraints[j][0].vector()[:]
                        )
                    )[0]
                )
                self.idx_active_upper.append(
                    np.nonzero(
                        (
                            self.controls[j].vector()[:]
                            >= self.control_constraints[j][1].vector()[:]
                        )
                    )[0]
                )
                self.idx_inactive.append(
                    np.nonzero(
                        np.logical_and(
                            self.controls[j].vector()[:]
                            > self.control_constraints[j][0].vector()[:],
                            self.controls[j].vector()[:]
                            < self.control_constraints[j][1].vector()[:],
                        )
                    )[0]
                )
            else:
                self.idx_active_lower.append([])
                self.idx_active_upper.append([])
                self.idx_inactive.append([])

            temp_active = np.concatenate(
                (self.idx_active_lower[j], self.idx_active_upper[j])
            )
            temp_active.sort()
            self.idx_active.append(temp_active)

    def restrict_to_active_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the active set.

        Restricts a control type function ``a`` onto the active set,
        which is returned via the function ``b``,  i.e., ``b`` is zero on the inactive
        set.

        Args:
            a: The first argument, to be projected onto the active set.
            b: The second argument, which stores the result (is overwritten).

        Returns:
            The result of the projection (overwrites input b).

        """
        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector().apply("")
                self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[
                    self.idx_active[j]
                ]
                self.temp[j].vector().apply("")
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())
                b[j].vector().apply("")

            else:
                b[j].vector().vec().set(0.0)
                b[j].vector().apply("")

        return b

    def restrict_to_inactive_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the inactive set.

        Restricts a control type function ``a`` onto the inactive set,
        which is returned via the function ``b``, i.e., ``b`` is zero on the active set.

        Args:
            a: The control-type function that is to be projected onto the inactive set.
            b: The storage for the result of the projection (is overwritten).

        Returns:
            The result of the projection of ``a`` onto the inactive set (overwrites
            input ``b``).

        """
        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector().apply("")
                self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[
                    self.idx_inactive[j]
                ]
                self.temp[j].vector().apply("")
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())
                b[j].vector().apply("")

            else:
                if not b[j].vector().vec().equal(a[j].vector().vec()):
                    b[j].vector().vec().aypx(0.0, a[j].vector().vec())
                    b[j].vector().apply("")

        return b

    def project_to_admissible_set(
        self, a: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Project a function to the set of admissible controls.

        Projects a control type function ``a`` onto the set of admissible controls
        (given by the box constraints).

        Args:
            a: The function which is to be projected onto the set of admissible
                controls (is overwritten)

        Returns:
            The result of the projection (overwrites input ``a``)

        """
        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                a[j].vector().vec().pointwiseMin(
                    self.control_constraints[j][1].vector().vec(), a[j].vector().vec()
                )
                a[j].vector().apply("")
                a[j].vector().vec().pointwiseMax(
                    a[j].vector().vec(), self.control_constraints[j][0].vector().vec()
                )
                a[j].vector().apply("")

        return a

    def _compute_gradient_equations(self) -> None:
        """Calculates the variational form of the gradient equation."""
        self.gradient_forms_rhs = [
            self.lagrangian.derivative(self.controls[i], self.test_functions_control[i])
            for i in range(self.control_dim)
        ]

    def _compute_sensitivity_equations(self) -> None:
        """Calculates the forms for the (forward) sensitivity equations."""
        # Use replace -> derivative to speed up the computations
        self.sensitivity_eqs_temp = [
            ufl.replace(
                self.state_forms[i], {self.adjoints[i]: self.test_functions_state[i]}
            )
            for i in range(self.state_dim)
        ]

        self.sensitivity_eqs_lhs: List[ufl.Form] = [
            fenics.derivative(
                self.sensitivity_eqs_temp[i],
                self.states[i],
                self.trial_functions_state[i],
            )
            for i in range(self.state_dim)
        ]
        if self.state_is_picard:
            self.sensitivity_eqs_picard = [
                fenics.derivative(
                    self.sensitivity_eqs_temp[i], self.states[i], self.states_prime[i]
                )
                for i in range(self.state_dim)
            ]

        # Need to distinguish cases due to empty sum in case state_dim = 1
        if self.state_dim > 1:
            # pylint: disable=invalid-unary-operand-type
            self.sensitivity_eqs_rhs: List[ufl.Form] = [
                -_utils.summation(
                    [
                        fenics.derivative(
                            self.sensitivity_eqs_temp[i],
                            self.states[j],
                            self.states_prime[j],
                        )
                        for j in range(self.state_dim)
                        if j != i
                    ]
                )
                - _utils.summation(
                    [
                        fenics.derivative(
                            self.sensitivity_eqs_temp[i],
                            self.controls[j],
                            self.test_directions[j],
                        )
                        for j in range(self.control_dim)
                    ]
                )
                for i in range(self.state_dim)
            ]
        else:
            self.sensitivity_eqs_rhs = [
                # pylint: disable=invalid-unary-operand-type
                -_utils.summation(
                    [
                        fenics.derivative(
                            self.sensitivity_eqs_temp[i],
                            self.controls[j],
                            self.test_directions[j],
                        )
                        for j in range(self.control_dim)
                    ]
                )
                for i in range(self.state_dim)
            ]

        # Add the right-hand-side for the picard iteration
        if self.state_is_picard:
            for i in range(self.state_dim):
                self.sensitivity_eqs_picard[i] -= self.sensitivity_eqs_rhs[i]

    def _compute_first_order_lagrangian_derivatives(self) -> None:
        """Computes the derivative of the Lagrangian w.r.t. the state and control."""
        self.lagrangian_y = [
            self.lagrangian.derivative(self.states[i], self.test_functions_state[i])
            for i in range(self.state_dim)
        ]
        self.lagrangian_u = [
            self.lagrangian.derivative(self.controls[i], self.test_functions_control[i])
            for i in range(self.control_dim)
        ]

    def _compute_second_order_lagrangian_derivatives(self) -> None:
        """Compute the second order derivatives of the Lagrangian w.r.t. y and u."""
        self.lagrangian_yy = [
            [
                fenics.derivative(
                    self.lagrangian_y[i], self.states[j], self.states_prime[j]
                )
                for j in range(self.state_dim)
            ]
            for i in range(self.state_dim)
        ]
        self.lagrangian_yu = [
            [
                fenics.derivative(
                    self.lagrangian_u[i], self.states[j], self.states_prime[j]
                )
                for j in range(self.state_dim)
            ]
            for i in range(self.control_dim)
        ]
        self.lagrangian_uy = [
            [
                fenics.derivative(
                    self.lagrangian_y[i], self.controls[j], self.test_directions[j]
                )
                for j in range(self.control_dim)
            ]
            for i in range(self.state_dim)
        ]
        self.lagrangian_uu = [
            [
                fenics.derivative(
                    self.lagrangian_u[i], self.controls[j], self.test_directions[j]
                )
                for j in range(self.control_dim)
            ]
            for i in range(self.control_dim)
        ]

    def _compute_adjoint_sensitivity_equations(self) -> None:
        """Computes the adjoint sensitivity equations for the Newton method."""
        # Use replace -> derivative for faster computations
        self.adjoint_sensitivity_eqs_diag_temp = [
            ufl.replace(
                self.state_forms[i], {self.adjoints[i]: self.trial_functions_adjoint[i]}
            )
            for i in range(self.state_dim)
        ]

        mapping_dict = {
            self.adjoints[j]: self.adjoints_prime[j] for j in range(self.state_dim)
        }
        self.adjoint_sensitivity_eqs_all_temp = [
            ufl.replace(self.state_forms[i], mapping_dict)
            for i in range(self.state_dim)
        ]

        self.adjoint_sensitivity_eqs_lhs = [
            fenics.derivative(
                self.adjoint_sensitivity_eqs_diag_temp[i],
                self.states[i],
                self.test_functions_adjoint[i],
            )
            for i in range(self.state_dim)
        ]
        if self.state_is_picard:
            self.adjoint_sensitivity_eqs_picard = [
                fenics.derivative(
                    self.adjoint_sensitivity_eqs_all_temp[i],
                    self.states[i],
                    self.test_functions_adjoint[i],
                )
                for i in range(self.state_dim)
            ]

        # Need cases distinction due to empty sum for state_dim == 1
        if self.state_dim > 1:
            for i in range(self.state_dim):
                self.w_1[i] -= _utils.summation(
                    [
                        fenics.derivative(
                            self.adjoint_sensitivity_eqs_all_temp[j],
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )
                        for j in range(self.state_dim)
                        if j != i
                    ]
                )
        else:
            pass

        # Add right-hand-side for picard iteration
        if self.state_is_picard:
            for i in range(self.state_dim):
                self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]

        self.adjoint_sensitivity_eqs_rhs = [
            _utils.summation(
                [
                    fenics.derivative(
                        self.adjoint_sensitivity_eqs_all_temp[j],
                        self.controls[i],
                        self.test_functions_control[i],
                    )
                    for j in range(self.state_dim)
                ]
            )
            for i in range(self.control_dim)
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
            for functional in self.cost_functional_list
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
            _utils.summation([self.lagrangian_yy[i][j] for j in range(self.state_dim)])
            + _utils.summation(
                [self.lagrangian_uy[i][j] for j in range(self.control_dim)]
            )
            for i in range(self.state_dim)
        ]
        self.w_2 = [
            _utils.summation([self.lagrangian_yu[i][j] for j in range(self.state_dim)])
            + _utils.summation(
                [self.lagrangian_uu[i][j] for j in range(self.control_dim)]
            )
            for i in range(self.control_dim)
        ]

        self._compute_adjoint_sensitivity_equations()

        self.w_3 = []
        self.hessian_rhs = []
        for i in range(self.control_dim):
            w3_i = -self.adjoint_sensitivity_eqs_rhs[i]
            self.w_3.append(w3_i)
            self.hessian_rhs.append(self.w_2[i] + self.w_3[i])
