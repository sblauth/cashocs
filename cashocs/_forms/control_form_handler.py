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

from typing import List, TYPE_CHECKING

import fenics
import numpy as np
from ufl import replace

from .form_handler import FormHandler
from .._exceptions import (
    InputError,
)
from ..utils import (
    summation,
    _max,
    _min,
)


if TYPE_CHECKING:
    from .._optimal_control.optimal_control_problem import OptimalControlProblem


class ControlFormHandler(FormHandler):
    """Class for UFL form manipulation for optimal control problems.

    This is used to symbolically derive the corresponding weak forms of the
    adjoint and gradient equations (via UFL) , that are later used in the
    solvers for the equations later on. These are needed as subroutines for
     the optimization (solution) algorithms.

    See Also
    --------
    ShapeFormHandler : Derives the adjoint equations and shape derivatives for shape optimization problems
    """

    def __init__(self, optimization_problem: OptimalControlProblem) -> None:
        """Initializes the ControlFormHandler class.

        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The corresponding optimal control problem
        """

        super().__init__(optimization_problem)

        # Initialize the attributes from the arguments
        self.controls = optimization_problem.controls
        self.riesz_scalar_products = optimization_problem.riesz_scalar_products
        self.control_constraints = optimization_problem.control_constraints
        self.require_control_constraints = (
            optimization_problem.require_control_constraints
        )

        self.control_dim = len(self.controls)
        self.control_spaces = [x.function_space() for x in self.controls]

        self.gradient = [fenics.Function(V) for V in self.control_spaces]

        # Define the necessary functions
        self.states_prime = [fenics.Function(V) for V in self.state_spaces]
        self.adjoints_prime = [fenics.Function(V) for V in self.adjoint_spaces]

        self.test_directions = [fenics.Function(V) for V in self.control_spaces]

        self.trial_functions_control = [
            fenics.TrialFunction(V) for V in self.control_spaces
        ]
        self.test_functions_control = [
            fenics.TestFunction(V) for V in self.control_spaces
        ]

        self.temp = [fenics.Function(V) for V in self.control_spaces]

        # Compute the necessary equations
        self.__compute_gradient_equations()

        if self.opt_algo == "newton" or (
            self.opt_algo == "pdas" and self.inner_pdas == "newton"
        ):
            self.__compute_newton_forms()

        # Initialize the scalar products
        fenics_scalar_product_matrices = [
            fenics.PETScMatrix() for i in range(self.control_dim)
        ]
        [
            fenics.assemble(
                self.riesz_scalar_products[i],
                keep_diagonal=True,
                tensor=fenics_scalar_product_matrices[i],
            )
            for i in range(self.control_dim)
        ]
        [
            fenics_scalar_product_matrices[i].ident_zeros()
            for i in range(self.control_dim)
        ]
        self.riesz_projection_matrices = [
            fenics_scalar_product_matrices[i].mat() for i in range(self.control_dim)
        ]

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
                        raise InputError(
                            "cashocs._forms.ControlFormHandler",
                            "riesz_scalar_products",
                            "Supplied scalar product form is not symmetric.",
                        )

    def scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between control type functions a and b.

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

        result = 0.0

        for i in range(self.control_dim):
            x = fenics.as_backend_type(a[i].vector()).vec()
            y = fenics.as_backend_type(b[i].vector()).vec()

            temp, _ = self.riesz_projection_matrices[i].getVecs()
            self.riesz_projection_matrices[i].mult(x, temp)
            result += temp.dot(y)

        return result

    def compute_active_sets(self) -> None:
        """Computes the indices corresponding to active and inactive sets.

        Returns
        -------
        None
        """

        self.idx_active_lower = []
        self.idx_active_upper = []
        self.idx_active = []
        self.idx_inactive = []

        for j in range(self.control_dim):

            if self.require_control_constraints[j]:
                self.idx_active_lower.append(
                    (
                        self.controls[j].vector()[:]
                        <= self.control_constraints[j][0].vector()[:]
                    ).nonzero()[0]
                )
                self.idx_active_upper.append(
                    (
                        self.controls[j].vector()[:]
                        >= self.control_constraints[j][1].vector()[:]
                    ).nonzero()[0]
                )
            else:
                self.idx_active_lower.append([])
                self.idx_active_upper.append([])

            temp_active = np.concatenate(
                (self.idx_active_lower[j], self.idx_active_upper[j])
            )
            temp_active.sort()
            self.idx_active.append(temp_active)
            self.idx_inactive.append(
                np.setdiff1d(
                    np.arange(self.control_spaces[j].dim()), self.idx_active[j]
                )
            )

    def restrict_to_active_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the active set.

        Restricts a control type function a onto the active set,
        which is returned via the function b,  i.e., b is zero on the inactive set.

        Parameters
        ----------
        a : list[fenics.Function]
            The first argument, to be projected onto the active set.
        b : list[fenics.Function]
            The second argument, which stores the result (is overwritten).

        Returns
        -------
        b : list[fenics.Function]
            The result of the projection (overwrites input b).
        """

        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[
                    self.idx_active[j]
                ]
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())

            else:
                b[j].vector().vec().set(0.0)

        return b

    def restrict_to_lower_active_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the lower bound of the constraints

        Parameters
        ----------
        a : list[fenics.Function]
            The input, which is to be restricted
        b : list[fenics.Function]
            The output, which stores the result

        Returns
        -------
        b : list[fenics.Function]
            Function a restricted onto the lower boundaries of the constraints

        """

        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector()[self.idx_active_lower[j]] = a[j].vector()[
                    self.idx_active_lower[j]
                ]
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())

            else:
                b[j].vector().vec().set(0.0)

        return b

    def restrict_to_upper_active_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the upper bound of the constraints

        Parameters
        ----------
        a : list[fenics.Function]
            The input, which is to be restricted
        b : list[fenics.Function]
            The output, which stores the result

        Returns
        -------
        b : list[fenics.Function]
            Function a restricted onto the upper boundaries of the constraints

        """

        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector()[self.idx_active_upper[j]] = a[j].vector()[
                    self.idx_active_upper[j]
                ]
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())

            else:
                b[j].vector().vec().set(0.0)

        return b

    def restrict_to_inactive_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the inactive set.

        Restricts a control type function a onto the inactive set,
        which is returned via the function b, i.e., b is zero on the active set.

        Parameters
        ----------
        a : list[fenics.Function]
            The control-type function that is to be projected onto the inactive set.
        b : list[fenics.Function]
            The storage for the result of the projection (is overwritten).

        Returns
        -------
        b : list[fenics.Function]
            The result of the projection of a onto the inactive set (overwrites input b).
        """

        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[
                    self.idx_inactive[j]
                ]
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())

            else:
                if not b[j].vector().vec().equal(a[j].vector().vec()):
                    b[j].vector().vec().aypx(0.0, a[j].vector().vec())

        return b

    def project_to_admissible_set(
        self, a: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Project a function to the set of admissible controls.

        Projects a control type function a onto the set of admissible controls
        (given by the box constraints).

        Parameters
        ----------
        a : list[fenics.Function]
            The function which is to be projected onto the set of admissible
            controls (is overwritten)

        Returns
        -------
        a : list[fenics.Function]
            The result of the projection (overwrites input a)
        """

        for j in range(self.control_dim):
            if self.require_control_constraints[j]:
                a[j].vector().vec().pointwiseMin(
                    self.control_constraints[j][1].vector().vec(), a[j].vector().vec()
                )
                a[j].vector().vec().pointwiseMax(
                    a[j].vector().vec(), self.control_constraints[j][0].vector().vec()
                )

        return a

    def __compute_gradient_equations(self) -> None:
        """Calculates the variational form of the gradient equation, for the Riesz projection.

        Returns
        -------
        None
        """

        self.gradient_forms_rhs = [
            fenics.derivative(
                self.lagrangian_form,
                self.controls[i],
                self.test_functions_control[i],
            )
            for i in range(self.control_dim)
        ]

        if self.use_scalar_tracking:
            for i in range(self.control_dim):
                for j in range(self.no_scalar_tracking_terms):
                    self.gradient_forms_rhs[i] += (
                        self.scalar_weights[j]
                        * (
                            self.scalar_cost_functional_integrand_values[j]
                            - fenics.Constant(self.scalar_tracking_goals[j])
                        )
                        * fenics.derivative(
                            self.scalar_cost_functional_integrands[j],
                            self.controls[i],
                            self.test_functions_control[i],
                        )
                    )

        if self.use_min_max_terms:
            for i in range(self.control_dim):
                for j in range(self.no_min_max_terms):
                    if self.min_max_lower_bounds[j] is not None:
                        term_lower = self.min_max_lambda[j] + self.min_max_mu[j] * (
                            self.min_max_integrand_values[j]
                            - self.min_max_lower_bounds[j]
                        )
                        self.gradient_forms_rhs[i] += _min(
                            fenics.Constant(0.0), term_lower
                        ) * fenics.derivative(
                            self.min_max_integrands[j],
                            self.controls[i],
                            self.test_functions_control[i],
                        )

                    if self.min_max_upper_bounds[j] is not None:
                        term_upper = self.min_max_lambda[j] + self.min_max_mu[j] * (
                            self.min_max_integrand_values[j]
                            - self.min_max_upper_bounds[j]
                        )
                        self.gradient_forms_rhs[i] += _max(
                            fenics.Constant(0.0), term_upper
                        ) * fenics.derivative(
                            self.min_max_integrands[j],
                            self.controls[i],
                            self.test_functions_control[i],
                        )

    def __compute_newton_forms(self) -> None:
        """Calculates the needed forms for the truncated Newton method.

        Returns
        -------
        None
        """

        if self.use_scalar_tracking or self.use_min_max_terms:
            raise InputError(
                "cashocs._forms.ShapeFormHandler",
                "__compute_newton_forms",
                "Newton's method is not available with scalar tracking or min_max terms.",
            )

        # Use replace -> derivative to speed up the computations
        self.sensitivity_eqs_temp = [
            replace(
                self.state_forms[i], {self.adjoints[i]: self.test_functions_state[i]}
            )
            for i in range(self.state_dim)
        ]

        self.sensitivity_eqs_lhs = [
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
            self.sensitivity_eqs_rhs = [
                -summation(
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
                - summation(
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
                -summation(
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

        # Compute forms for the truncated Newton method
        self.L_y = [
            fenics.derivative(
                self.lagrangian_form,
                self.states[i],
                self.test_functions_state[i],
            )
            for i in range(self.state_dim)
        ]
        self.L_u = [
            fenics.derivative(
                self.lagrangian_form,
                self.controls[i],
                self.test_functions_control[i],
            )
            for i in range(self.control_dim)
        ]

        self.L_yy = [
            [
                fenics.derivative(self.L_y[i], self.states[j], self.states_prime[j])
                for j in range(self.state_dim)
            ]
            for i in range(self.state_dim)
        ]
        self.L_yu = [
            [
                fenics.derivative(self.L_u[i], self.states[j], self.states_prime[j])
                for j in range(self.state_dim)
            ]
            for i in range(self.control_dim)
        ]
        self.L_uy = [
            [
                fenics.derivative(
                    self.L_y[i], self.controls[j], self.test_directions[j]
                )
                for j in range(self.control_dim)
            ]
            for i in range(self.state_dim)
        ]
        self.L_uu = [
            [
                fenics.derivative(
                    self.L_u[i], self.controls[j], self.test_directions[j]
                )
                for j in range(self.control_dim)
            ]
            for i in range(self.control_dim)
        ]

        self.w_1 = [
            summation([self.L_yy[i][j] for j in range(self.state_dim)])
            + summation([self.L_uy[i][j] for j in range(self.control_dim)])
            for i in range(self.state_dim)
        ]
        self.w_2 = [
            summation([self.L_yu[i][j] for j in range(self.state_dim)])
            + summation([self.L_uu[i][j] for j in range(self.control_dim)])
            for i in range(self.control_dim)
        ]

        # Use replace -> derivative for faster computations
        self.adjoint_sensitivity_eqs_diag_temp = [
            replace(
                self.state_forms[i], {self.adjoints[i]: self.trial_functions_adjoint[i]}
            )
            for i in range(self.state_dim)
        ]

        mapping_dict = {
            self.adjoints[j]: self.adjoints_prime[j] for j in range(self.state_dim)
        }
        self.adjoint_sensitivity_eqs_all_temp = [
            replace(self.state_forms[i], mapping_dict) for i in range(self.state_dim)
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
                self.w_1[i] -= summation(
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
            summation(
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

        self.w_3 = [
            -self.adjoint_sensitivity_eqs_rhs[i] for i in range(self.control_dim)
        ]

        self.hessian_rhs = [self.w_2[i] + self.w_3[i] for i in range(self.control_dim)]
