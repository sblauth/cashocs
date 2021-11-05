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


from .constraints import EqualityConstraint, InequalityConstraint
import fenics


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

        self.iterations = 0
        self.constraint_violation = 0.0
        self.constraint_violation_prev = 0.0

    def solve(self, tol=1e-2, max_iter=10):
        pass

    def total_constraint_violation(self):
        s = 0.0
        for constraint in self.constraints:
            s += constraint.constraint_violation()

        return s


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
        for i, constraint in enumerate(self.constraints):
            if constraint.is_integral_constraint:
                self.cost_functional_form = self.cost_functional_form_initial + [
                    fenics.Constant(self.lmbd[i]) * constraint.linear_term
                ]
                constraint.quadratic_term["weight"] = self.mu
                if self.scalar_tracking_forms_initial is not None:
                    self.scalar_tracking_forms = self.scalar_tracking_forms_initial + [
                        constraint.quadratic_term
                    ]
                else:
                    self.scalar_tracking_forms = [constraint.quadratic_term]

            elif constraint.is_pointwise_constraint:
                self.cost_functional_form = self.cost_functional_form_initial + [
                    fenics.Constant(self.lmbd[i]) * constraint.linear_term,
                    fenics.Constant(self.mu) * constraint.quadratic_term,
                ]

    def _update_lagrange_multiplier_estimates(self):
        for i in range(self.constraint_dim):
            if isinstance(self.constraints[i], EqualityConstraint):
                if self.constraints[i].is_integral_constraint:
                    self.lmbd[i] += self.mu * (
                        fenics.assemble(self.constraints[i].variable_function)
                        - self.constraints[i].target
                    )
            elif isinstance(self.constraints[i], InequalityConstraint):
                if self.constraints[i].is_integral_constraint:
                    pass

    def solve(self, tol=1e-2, max_iter=10):
        pass


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
