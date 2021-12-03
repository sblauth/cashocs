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

import abc

import numpy as np

from .solvers import AugmentedLagrangianMethod, QuadraticPenaltyMethod
from .._exceptions import InputError
from .._optimal_control.optimal_control_problem import OptimalControlProblem
from .._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from ..utils import enlist


class ConstrainedOptimizationProblem(abc.ABC):
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
        scalar_tracking_forms=None,
    ):

        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.states = states
        self.adjoints = adjoints
        self.config = config
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options

        self.solver = None

        self.cost_functional_form_initial = enlist(cost_functional_form)
        if scalar_tracking_forms is not None:
            self.scalar_tracking_forms_initial = enlist(scalar_tracking_forms)
        else:
            self.scalar_tracking_forms_initial = None
        self.constraints = enlist(constraints)

        self.constraint_dim = len(self.constraints)

        self.iterations = 0
        self.initial_norm = 1.0
        self.constraint_violation = 0.0
        self.constraint_violation_prev = 0.0

        self.cost_functional_shift = 0.0

    def solve(
        self,
        method="Augmented Lagrangian",
        tol=1e-2,
        max_iter=25,
        inner_rtol=None,
        inner_atol=None,
        constraint_tol=None,
        mu_0=None,
        lambda_0=None,
    ):
        if method in ["Augmented Lagrangian", "AL"]:
            self.solver = AugmentedLagrangianMethod(self, mu_0=mu_0, lambda_0=lambda_0)
        elif method in ["Quadratic Penalty", "QP"]:
            self.solver = QuadraticPenaltyMethod(self, mu_0=mu_0, lambda_0=lambda_0)
        else:
            raise InputError(
                "cashocs._constraints.constrained_problems.ConstrainedOptimizationProblem.solve",
                "method",
                "The parameter `method` should be either 'AL' or 'Augmented Lagrangian' or 'QP' or 'Quadratic Penalty'",
            )

        self.solver.solve(
            tol=tol,
            max_iter=max_iter,
            inner_rtol=inner_rtol,
            inner_atol=inner_atol,
            constraint_tol=constraint_tol,
        )

    def total_constraint_violation(self):
        s = 0.0
        for constraint in self.constraints:
            s += pow(constraint.constraint_violation(), 2)

        return np.sqrt(s)

    @abc.abstractmethod
    def _solve_inner_problem(self, tol=1e-2, inner_rtol=None, inner_atol=None):
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


class ConstrainedOptimalControlProblem(ConstrainedOptimizationProblem):
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        controls,
        adjoints,
        constraints,
        config=None,
        riesz_scalar_products=None,
        control_constraints=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        scalar_tracking_forms=None,
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
            scalar_tracking_forms=scalar_tracking_forms,
        )

        self.controls = controls
        self.riesz_scalar_products = riesz_scalar_products
        self.control_constraints = control_constraints

    def _solve_inner_problem(self, tol=1e-2, inner_rtol=None, inner_atol=None):
        ocp = OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.solver.inner_cost_functional_form,
            self.states,
            self.controls,
            self.adjoints,
            config=self.config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            scalar_tracking_forms=self.solver.inner_scalar_tracking_forms,
            min_max_terms=self.solver.inner_min_max_terms,
        )

        ocp.inject_pre_post_hook(self._pre_hook, self._post_hook)
        ocp._OptimizationProblem__shift_cost_functional(
            self.solver.inner_cost_functional_shift
        )

        if inner_rtol is not None:
            rtol = inner_rtol
        else:
            rtol = tol

        if inner_atol is not None:
            atol = inner_atol
        else:
            if self.iterations == 1:
                ocp.compute_gradient()
                self.initial_norm = np.sqrt(ocp.gradient_problem.gradient_norm_squared)
            atol = self.initial_norm * tol / 10.0

        ocp.solve(rtol=rtol, atol=atol)


class ConstrainedShapeOptimizationProblem(ConstrainedOptimizationProblem):
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        adjoints,
        boundaries,
        constraints,
        config=None,
        shape_scalar_product=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        scalar_tracking_forms=None,
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
            scalar_tracking_forms=scalar_tracking_forms,
        )

        self.boundaries = boundaries
        self.shape_scalar_product = shape_scalar_product

    def _solve_inner_problem(self, tol=1e-2, inner_rtol=None, inner_atol=None):
        sop = ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.solver.inner_cost_functional_form,
            self.states,
            self.adjoints,
            self.boundaries,
            config=self.config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            scalar_tracking_forms=self.solver.inner_scalar_tracking_forms,
            min_max_terms=self.solver.inner_min_max_terms,
        )
        sop.inject_pre_post_hook(self._pre_hook, self._post_hook)
        sop._OptimizationProblem__shift_cost_functional(
            self.solver.inner_cost_functional_shift
        )

        if inner_rtol is not None:
            rtol = inner_rtol
        else:
            rtol = tol

        if inner_atol is not None:
            atol = inner_atol
        else:
            if self.iterations == 1:
                sop.compute_shape_gradient()
                self.initial_norm = np.sqrt(
                    sop.shape_gradient_problem.gradient_norm_squared
                )
            atol = self.initial_norm * tol / 10.0

        sop.solve(rtol=rtol, atol=atol)
