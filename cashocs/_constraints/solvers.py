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


from .._loggers import debug
from .constraints import EqualityConstraint, InequalityConstraint
from ..utils import _max, _min, summation, _assemble_petsc_system, _solve_linear_problem
import fenics
import numpy as np


class ConstrainedOptimizationProblem:
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        adjoints,
        constraints,
        config=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        desired_weights=None,
        scalar_tracking_forms=None,
        mu_0=None,
        lambda_0=None,
    ):

        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.states = states
        self.adjoints = adjoints
        self.config = config
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.desired_weights = desired_weights

        if not isinstance(cost_functional_form, list):
            self.cost_functional_form_initial = [cost_functional_form]
        else:
            self.cost_functional_form_initial = cost_functional_form

        if scalar_tracking_forms is not None:
            if not isinstance(scalar_tracking_forms, list):
                self.scalar_tracking_forms_initial = [scalar_tracking_forms]
            else:
                self.scalar_tracking_forms_initial = scalar_tracking_forms
        else:
            self.scalar_tracking_forms_initial = None

        if not isinstance(constraints, list):
            self.constraints = [constraints]
        else:
            self.constraints = constraints
        self.constraint_dim = len(self.constraints)

        if mu_0 is not None:
            self.mu = mu_0
        else:
            self.mu = 1.0

        if lambda_0 is not None:
            if not isinstance(lambda_0, list):
                self.lmbd = [lambda_0]
            else:
                self.lmbd = lambda_0
        else:
            self.lmbd = [0.0] * self.constraint_dim

        for constraint in self.constraints:
            if constraint.is_pointwise_constraint:
                mesh = constraint.measure.ufl_domain().ufl_cargo()
                self.CG = fenics.FunctionSpace(mesh, "CG", 1)
                break

        for i, constraint in enumerate(self.constraints):
            if constraint.is_pointwise_constraint:
                constraint.multiplier.vector()[:] = self.lmbd[i]
                self.lmbd[i] = constraint.multiplier

        self.iterations = 0
        self.constraint_violation = 0.0
        self.constraint_violation_prev = 0.0

    def solve(self, tol=1e-2, max_iter=10):
        pass

    def total_constraint_violation(self):
        s = 0.0
        for constraint in self.constraints:
            s += pow(constraint.constraint_violation(), 2)

        return np.sqrt(s)

    def _solve_inner_problem(self, tol=1e-2):
        pass

    def _pre_hook(self):
        pass

    def _post_hook(self):
        pass

    def inject_pre_hook(self, function):
        """
        Changes the a-priori hook of the OptimizationProblem

        Parameters
        ----------
        function : function
            A custom function without arguments, which will be called before each solve
            of the state system

        Returns
        -------
         : None

        """

        self._pre_hook = function

    def inject_post_hook(self, function):
        """
        Changes the a-posteriori hook of the OptimizationProblem

        Parameters
        ----------
        function : function
            A custom function without arguments, which will be called after the computation
            of the gradient(s)

        Returns
        -------
         : None

        """

        self._post_hook = function

    def inject_pre_post_hook(self, pre_function, post_function):
        """
        Changes the a-priori (pre) and a-posteriori (post) hook of the OptimizationProblem

        Parameters
        ----------
        pre_function : function
            A function without arguments, which is to be called before each solve of the
            state system
        post_function : function
            A function without arguments, which is to be called after each computation of
            the (shape) gradient

        Returns
        -------
         : None

        """

        self.inject_pre_hook(pre_function)
        self.inject_post_hook(post_function)


class AugmentedLagrangianProblem(ConstrainedOptimizationProblem):
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        adjoints,
        constraints,
        config=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        desired_weights=None,
        scalar_tracking_forms=None,
        mu_0=None,
        lambda_0=None,
    ):
        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            constraints,
            config=config,
            initial_guess=initial_guess,
            ksp_options=ksp_options,
            adjoint_ksp_options=adjoint_ksp_options,
            desired_weights=desired_weights,
            scalar_tracking_forms=scalar_tracking_forms,
            mu_0=mu_0,
            lambda_0=lambda_0,
        )
        self.gamma = 0.25
        self.beta = 10.0

    def _update_cost_functional(self):
        has_scalar_tracking_terms = False
        has_min_max_terms = False

        self.cost_functional_form = self.cost_functional_form_initial
        if self.scalar_tracking_forms_initial is not None:
            self.scalar_tracking_forms = self.scalar_tracking_forms_initial
        else:
            self.scalar_tracking_forms = []
        self.min_max_terms = []

        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint, EqualityConstraint):
                if constraint.is_integral_constraint:
                    has_scalar_tracking_terms = True
                    self.cost_functional_form += [
                        fenics.Constant(self.lmbd[i]) * constraint.linear_term
                    ]
                    constraint.quadratic_term["weight"] = self.mu
                    self.scalar_tracking_forms += [constraint.quadratic_term]

                elif constraint.is_pointwise_constraint:
                    self.cost_functional_form += [
                        constraint.linear_term,
                        fenics.Constant(self.mu) * constraint.quadratic_term,
                    ]

            elif isinstance(constraint, InequalityConstraint):
                if constraint.is_integral_constraint:
                    has_min_max_terms = True
                    constraint.min_max_term["mu"] = self.mu
                    constraint.min_max_term["lambda"] = self.lmbd[i]
                    self.min_max_terms += [constraint.min_max_term]

                elif constraint.is_pointwise_constraint:
                    constraint.weight.vector()[:] = self.mu
                    self.cost_functional_form += constraint.cost_functional_terms

            if not has_scalar_tracking_terms:
                self.scalar_tracking_forms = self.scalar_tracking_forms_initial
            if not has_min_max_terms:
                self.min_max_terms = None

    def project_pointwise_multiplier(self, project_terms, measure, index):
        if isinstance(project_terms, list):
            project_term = summation(project_terms)
        else:
            project_term = project_terms

        trial = fenics.TrialFunction(self.CG)
        test = fenics.TestFunction(self.CG)

        a = trial * test * measure
        L = project_term * test * measure

        A, b = _assemble_petsc_system(a, L)
        _solve_linear_problem(A=A, b=b, x=self.lmbd[index].vector().vec())

    def _update_lagrange_multiplier_estimates(self):
        for i in range(self.constraint_dim):
            if isinstance(self.constraints[i], EqualityConstraint):
                if self.constraints[i].is_integral_constraint:
                    self.lmbd[i] += self.mu * (
                        fenics.assemble(self.constraints[i].variable_function)
                        - self.constraints[i].target
                    )
                elif self.constraints[i].is_pointwise_constraint:
                    project_term = self.lmbd[i] + self.mu * (
                        self.constraints[i].variable_function
                        - self.constraints[i].target
                    )
                    self.project_pointwise_multiplier(
                        project_term, self.constraints[i].measure, i
                    )

            elif isinstance(self.constraints[i], InequalityConstraint):
                if self.constraints[i].is_integral_constraint:
                    lower_term = 0.0
                    upper_term = 0.0

                    min_max_integral = fenics.assemble(
                        self.constraints[i].min_max_term["integrand"]
                    )

                    if self.constraints[i].lower_bound is not None:
                        lower_term = np.minimum(
                            self.lmbd[i]
                            + self.mu
                            * (min_max_integral - self.constraints[i].lower_bound),
                            0.0,
                        )

                    if self.constraints[i].upper_bound is not None:
                        upper_term = np.maximum(
                            self.lmbd[i]
                            + self.mu
                            * (min_max_integral - self.constraints[i].upper_bound),
                            0.0,
                        )

                    self.lmbd[i] = lower_term + upper_term
                    self.constraints[i].min_max_term["lambda"] = self.lmbd[i]

                elif self.constraints[i].is_pointwise_constraint:
                    project_terms = []
                    if self.constraints[i].upper_bound is not None:
                        project_terms.append(
                            _max(
                                self.lmbd[i]
                                + self.mu
                                * (
                                    self.constraints[i].variable_function
                                    - self.constraints[i].upper_bound
                                ),
                                fenics.Constant(0.0),
                            )
                        )

                    if self.constraints[i].lower_bound is not None:
                        project_terms.append(
                            _min(
                                self.lmbd[i]
                                + self.mu
                                * (
                                    self.constraints[i].variable_function
                                    - self.constraints[i].lower_bound
                                ),
                                fenics.Constant(0.0),
                            )
                        )

                    self.project_pointwise_multiplier(
                        project_terms, self.constraints[i].measure, i
                    )

    def solve(self, tol=1e-2, max_iter=10):
        self.iterations = 0
        while True:
            self.iterations += 1

            debug(f"{self.mu = }")
            debug(f"{self.lmbd = }")

            self._update_cost_functional()

            self._solve_inner_problem(tol=tol)

            self._update_lagrange_multiplier_estimates()

            self.constraint_violation_prev = self.constraint_violation
            self.constraint_violation = self.total_constraint_violation()

            if self.constraint_violation > self.gamma * self.constraint_violation_prev:
                self.mu *= self.beta

            if self.constraint_violation <= tol / 10.0:
                print("Converged successfully.")
                break

            if self.iterations >= max_iter:
                print("Augmented Lagrangian did not converge")
                break


# class LagrangianProblem(ConstrainedOptimizationProblem):
#     def __init__(self, optimization_problem, constraints):
#         super().__init__(optimization_problem, constraints)
#         pass
#
#
# class QuadraticPenaltyProblem(ConstrainedOptimizationProblem):
#     def __init__(self, optimization_problem, constraints):
#         super().__init__(optimization_problem, constraints)
#         pass
#
#
# class L1PenaltyProblem(ConstrainedOptimizationProblem):
#     def __init__(self, optimization_problem, constraints):
#         super().__init__(optimization_problem, constraints)
