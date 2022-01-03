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

import abc
from typing import TYPE_CHECKING

import fenics
import numpy as np
from ufl import replace
from ufl.log import UFLException

from .._exceptions import (
    CashocsException,
)
from ..utils import (
    _optimization_algorithm_configuration,
    summation,
    _max,
    _min,
)


if TYPE_CHECKING:
    from .._interfaces.optimization_problem import OptimizationProblem


class FormHandler(abc.ABC):
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

        self.gradient = None

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

    @abc.abstractmethod
    def scalar_product(self, a, b) -> float:
        pass
