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

"""Private module forms of CASHOCS.

This is used to carry out form manipulations such as generating the UFL
 forms for the adjoint system and for the Riesz gradient identificiation
problems.
"""

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
from ufl.log import UFLException

from ._exceptions import (
    CashocsException,
    InputError,
    IncompatibleConfigurationError,
)
from ._loggers import warning
from ._shape_optimization.regularization import Regularization
from .geometry import compute_boundary_distance
from .utils import (
    _assemble_petsc_system,
    _optimization_algorithm_configuration,
    _setup_petsc_options,
    _solve_linear_problem,
    create_dirichlet_bcs,
    summation,
    _max,
    _min,
)


if TYPE_CHECKING:
    from ._interfaces.optimization_problem import OptimizationProblem
    from ._optimal_control.optimal_control_problem import OptimalControlProblem
    from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem


class FormHandler:
    """Parent class for UFL form manipulation.

    This is subclassed by specific form handlers for either
    optimal control or shape optimization. The base class is
    used to determine common objects and to derive the UFL forms
    for the state and adjoint systems.

    See Also
    --------
    ControlFormHandler : FormHandler for optimal control problems
    ShapeFormHandler : FormHandler for shape optimization problems
    """

    def __init__(self, optimization_problem: OptimizationProblem) -> None:
        """Initializes the form handler.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The corresponding optimization problem
        """

        self.bcs_list = optimization_problem.bcs_list
        self.states = optimization_problem.states
        self.adjoints = optimization_problem.adjoints
        self.config = optimization_problem.config
        self.state_ksp_options = optimization_problem.ksp_options
        self.adjoint_ksp_options = optimization_problem.adjoint_ksp_options
        self.use_scalar_tracking = optimization_problem.use_scalar_tracking
        self.use_min_max_terms = optimization_problem.use_min_max_terms
        self.min_max_forms = optimization_problem.min_max_terms
        self.scalar_tracking_forms = optimization_problem.scalar_tracking_forms

        self.cost_functional_form = optimization_problem.cost_functional_form
        self.state_forms = optimization_problem.state_forms

        self.lagrangian_form = self.cost_functional_form + summation(self.state_forms)
        self.cost_functional_shift = 0.0

        if self.use_scalar_tracking:
            self.scalar_cost_functional_integrands = [
                d["integrand"] for d in self.scalar_tracking_forms
            ]
            self.scalar_tracking_goals = [
                d["tracking_goal"] for d in self.scalar_tracking_forms
            ]

            dummy_meshes = [
                integrand.integrals()[0].ufl_domain().ufl_cargo()
                for integrand in self.scalar_cost_functional_integrands
            ]
            self.scalar_cost_functional_integrand_values = [
                fenics.Function(fenics.FunctionSpace(mesh, "R", 0))
                for mesh in dummy_meshes
            ]
            self.scalar_weights = [
                fenics.Function(fenics.FunctionSpace(mesh, "R", 0))
                for mesh in dummy_meshes
            ]

            self.no_scalar_tracking_terms = len(self.scalar_tracking_goals)
            try:
                for j in range(self.no_scalar_tracking_terms):
                    self.scalar_weights[j].vector().vec().set(
                        self.scalar_tracking_forms[j]["weight"]
                    )
            except KeyError:
                for j in range(self.no_scalar_tracking_terms):
                    self.scalar_weights[j].vector().vec().set(1.0)

        if self.use_min_max_terms:
            self.min_max_integrands = [d["integrand"] for d in self.min_max_forms]
            self.min_max_lower_bounds = [d["lower_bound"] for d in self.min_max_forms]
            self.min_max_upper_bounds = [d["upper_bound"] for d in self.min_max_forms]

            dummy_meshes = [
                integrand.integrals()[0].ufl_domain().ufl_cargo()
                for integrand in self.min_max_integrands
            ]
            self.min_max_integrand_values = [
                fenics.Function(fenics.FunctionSpace(mesh, "R", 0))
                for mesh in dummy_meshes
            ]

            self.no_min_max_terms = len(self.min_max_integrands)
            self.min_max_mu = []
            self.min_max_lambda = []
            for j in range(self.no_min_max_terms):
                self.min_max_mu.append(self.min_max_forms[j]["mu"])
                self.min_max_lambda.append(self.min_max_forms[j]["lambda"])

        self.state_dim = len(self.states)

        self.state_spaces = [x.function_space() for x in self.states]
        self.adjoint_spaces = [x.function_space() for x in self.adjoints]

        # Test if state_spaces coincide with adjoint_spaces
        if self.state_spaces == self.adjoint_spaces:
            self.state_adjoint_equal_spaces = True
        else:
            self.state_adjoint_equal_spaces = False

        self.mesh = self.state_spaces[0].mesh()
        self.dx = fenics.Measure("dx", self.mesh)

        self.trial_functions_state = [
            fenics.TrialFunction(V) for V in self.state_spaces
        ]
        self.test_functions_state = [fenics.TestFunction(V) for V in self.state_spaces]

        self.trial_functions_adjoint = [
            fenics.TrialFunction(V) for V in self.adjoint_spaces
        ]
        self.test_functions_adjoint = [
            fenics.TestFunction(V) for V in self.adjoint_spaces
        ]

        self.state_is_linear = self.config.getboolean(
            "StateSystem", "is_linear", fallback=False
        )
        self.state_is_picard = self.config.getboolean(
            "StateSystem", "picard_iteration", fallback=False
        )
        self.opt_algo = _optimization_algorithm_configuration(self.config)

        if self.opt_algo == "pdas":
            self.inner_pdas = self.config.get("AlgoPDAS", "inner_pdas")

        self.__compute_state_equations()
        self.__compute_adjoint_equations()

    def __compute_state_equations(self) -> None:
        """Calculates the weak form of the state equation for the use with fenics.

        Returns
        -------
        None
        """

        if self.state_is_linear:
            self.state_eq_forms = [
                replace(
                    self.state_forms[i],
                    {
                        self.states[i]: self.trial_functions_state[i],
                        self.adjoints[i]: self.test_functions_state[i],
                    },
                )
                for i in range(self.state_dim)
            ]

        else:
            self.state_eq_forms = [
                fenics.derivative(
                    self.state_forms[i], self.adjoints[i], self.test_functions_state[i]
                )
                for i in range(self.state_dim)
            ]

        if self.state_is_picard:
            self.state_picard_forms = [
                fenics.derivative(
                    self.state_forms[i], self.adjoints[i], self.test_functions_state[i]
                )
                for i in range(self.state_dim)
            ]

        if self.state_is_linear:
            self.state_eq_forms_lhs = []
            self.state_eq_forms_rhs = []
            for i in range(self.state_dim):
                try:
                    a, L = fenics.system(self.state_eq_forms[i])
                except UFLException:
                    raise CashocsException(
                        "The state system could not be transferred to a linear system.\n"
                        "Perhaps you specified that the system is linear, allthough it is not.\n"
                        "In your config, in the StateSystem section, try using is_linear = False."
                    )
                self.state_eq_forms_lhs.append(a)
                if L.empty():
                    zero_form = (
                        fenics.inner(
                            fenics.Constant(
                                np.zeros(self.test_functions_state[i].ufl_shape)
                            ),
                            self.test_functions_state[i],
                        )
                        * self.dx
                    )
                    self.state_eq_forms_rhs.append(zero_form)
                else:
                    self.state_eq_forms_rhs.append(L)

    def __compute_adjoint_equations(self) -> None:
        """Calculates the weak form of the adjoint equation for use with fenics.

        Returns
        -------
        None
        """

        # Use replace -> derivative to speed up computations
        self.lagrangian_temp_forms = [
            replace(
                self.lagrangian_form,
                {self.adjoints[i]: self.trial_functions_adjoint[i]},
            )
            for i in range(self.state_dim)
        ]

        if self.state_is_picard:
            self.adjoint_picard_forms = [
                fenics.derivative(
                    self.lagrangian_form,
                    self.states[i],
                    self.test_functions_adjoint[i],
                )
                for i in range(self.state_dim)
            ]

            if self.use_scalar_tracking:
                for i in range(self.state_dim):
                    for j in range(self.no_scalar_tracking_terms):
                        self.adjoint_picard_forms[i] += self.scalar_weights[j](
                            self.scalar_cost_functional_integrand_values[j]
                            - fenics.Constant(self.scalar_tracking_goals[j])
                        ) * fenics.derivative(
                            self.scalar_cost_functional_integrands[j],
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )

            if self.use_min_max_terms:
                for i in range(self.state_dim):
                    for j in range(self.no_min_max_terms):

                        if self.min_max_lower_bounds[j] is not None:
                            term_lower = self.min_max_lambda[j] + self.min_max_mu[j] * (
                                self.min_max_integrand_values[j]
                                - self.min_max_lower_bounds[j]
                            )
                            self.adjoint_picard_forms[i] += _min(
                                fenics.Constant(0.0), term_lower
                            ) * fenics.derivative(
                                self.min_max_integrands[j],
                                self.states[i],
                                self.test_functions_adjoint[i],
                            )

                        if self.min_max_upper_bounds[j] is not None:
                            term_upper = self.min_max_lambda[j] + self.min_max_mu[j] * (
                                self.min_max_integrand_values[j]
                                - self.min_max_upper_bounds[j]
                            )
                            self.adjoint_picard_forms[i] += _max(
                                fenics.Constant(0.0), term_upper
                            ) * fenics.derivative(
                                self.min_max_integrands[j],
                                self.states[i],
                                self.test_functions_adjoint[i],
                            )

        self.adjoint_eq_forms = [
            fenics.derivative(
                self.lagrangian_temp_forms[i],
                self.states[i],
                self.test_functions_adjoint[i],
            )
            for i in range(self.state_dim)
        ]
        if self.use_scalar_tracking:
            for i in range(self.state_dim):
                for j in range(self.no_scalar_tracking_terms):
                    self.temp_form = replace(
                        self.scalar_cost_functional_integrands[j],
                        {self.adjoints[i]: self.trial_functions_adjoint[i]},
                    )
                    self.adjoint_eq_forms[i] += (
                        self.scalar_weights[j]
                        * (
                            self.scalar_cost_functional_integrand_values[j]
                            - fenics.Constant(self.scalar_tracking_goals[j])
                        )
                        * fenics.derivative(
                            self.temp_form,
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )
                    )

        if self.use_min_max_terms:
            for i in range(self.state_dim):
                for j in range(self.no_min_max_terms):
                    self.temp_form = replace(
                        self.min_max_integrands[j],
                        {self.adjoints[i]: self.trial_functions_adjoint[i]},
                    )
                    if self.min_max_lower_bounds[j] is not None:
                        term_lower = self.min_max_lambda[j] + self.min_max_mu[j] * (
                            self.min_max_integrand_values[j]
                            - self.min_max_lower_bounds[j]
                        )
                        self.adjoint_eq_forms[i] += _min(
                            fenics.Constant(0.0), term_lower
                        ) * fenics.derivative(
                            self.temp_form,
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )

                    if self.min_max_upper_bounds[j] is not None:
                        term_upper = self.min_max_lambda[j] + self.min_max_mu[j] * (
                            self.min_max_integrand_values[j]
                            - self.min_max_upper_bounds[j]
                        )
                        self.adjoint_eq_forms[i] += _max(
                            fenics.Constant(0.0), term_upper
                        ) * fenics.derivative(
                            self.temp_form,
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )

        self.adjoint_eq_lhs = []
        self.adjoint_eq_rhs = []

        for i in range(self.state_dim):
            a, L = fenics.system(self.adjoint_eq_forms[i])
            self.adjoint_eq_lhs.append(a)
            if L.empty():
                zero_form = (
                    fenics.inner(
                        fenics.Constant(
                            np.zeros(self.test_functions_adjoint[i].ufl_shape)
                        ),
                        self.test_functions_adjoint[i],
                    )
                    * self.dx
                )
                self.adjoint_eq_rhs.append(zero_form)
            else:
                self.adjoint_eq_rhs.append(L)

        # Compute the  adjoint boundary conditions
        if self.state_adjoint_equal_spaces:
            self.bcs_list_ad = [
                [fenics.DirichletBC(bc) for bc in self.bcs_list[i]]
                for i in range(self.state_dim)
            ]
            [
                [bc.homogenize() for bc in self.bcs_list_ad[i]]
                for i in range(self.state_dim)
            ]
        else:

            def get_subdx(V, idx, ls):
                if V.id() == idx:
                    return ls
                if V.num_sub_spaces() > 1:
                    for i in range(V.num_sub_spaces()):
                        ans = get_subdx(V.sub(i), idx, ls + [i])
                        if ans is not None:
                            return ans
                else:
                    return None

            self.bcs_list_ad = [
                [1 for bc in range(len(self.bcs_list[i]))]
                for i in range(self.state_dim)
            ]

            for i in range(self.state_dim):
                for j, bc in enumerate(self.bcs_list[i]):
                    idx = bc.function_space().id()
                    subdx = get_subdx(self.state_spaces[i], idx, ls=[])
                    W = self.adjoint_spaces[i]
                    for num in subdx:
                        W = W.sub(num)
                    shape = W.ufl_element().value_shape()
                    try:
                        if shape == ():
                            self.bcs_list_ad[i][j] = fenics.DirichletBC(
                                W,
                                fenics.Constant(0),
                                bc.domain_args[0],
                                bc.domain_args[1],
                            )
                        else:
                            self.bcs_list_ad[i][j] = fenics.DirichletBC(
                                W,
                                fenics.Constant([0] * W.ufl_element().value_size()),
                                bc.domain_args[0],
                                bc.domain_args[1],
                            )
                    except AttributeError:
                        if shape == ():
                            self.bcs_list_ad[i][j] = fenics.DirichletBC(
                                W, fenics.Constant(0), bc.sub_domain
                            )
                        else:
                            self.bcs_list_ad[i][j] = fenics.DirichletBC(
                                W,
                                fenics.Constant([0] * W.ufl_element().value_size()),
                                bc.sub_domain,
                            )

    def _pre_hook(self) -> None:
        pass

    def _post_hook(self) -> None:
        pass


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

        self.gradient = fenics.Function(self.deformation_space)
        self.test_vector_field = fenics.TestFunction(self.deformation_space)

        self.regularization = Regularization(self)

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
        self.shape_derivative += self.regularization.compute_shape_derivative()

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

    def scalar_product(self, a: fenics.Function, b: fenics.Function) -> float:
        """Computes the scalar product between two deformation functions.

        Parameters
        ----------
        a : fenics.Function
            The first argument.
        b : fenics.Function
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
                self.F_p_laplace, {self.gradient: a, self.test_vector_field: b}
            )
            result = fenics.assemble(self.form)

        else:
            x = fenics.as_backend_type(a.vector()).vec()
            y = fenics.as_backend_type(b.vector()).vec()

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
                fenics.inner(fenics.grad(self.gradient), fenics.grad(self.gradient)),
                (p - 2) / 2.0,
            )
            self.F_p_laplace = (
                fenics.inner(
                    self.mu_lame
                    * (fenics.Constant(eps) + kappa)
                    * fenics.grad(self.gradient),
                    fenics.grad(self.test_vector_field),
                )
                * self.dx
                + fenics.Constant(delta)
                * fenics.dot(self.gradient, self.test_vector_field)
                * self.dx
            )
