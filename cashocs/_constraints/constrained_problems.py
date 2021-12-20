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

"""Constrained Optimization Problems, with additional equality and inequality constraints.

"""

from __future__ import annotations

import abc
import configparser
from typing import List, Dict, Optional, Union, Callable

import fenics
import numpy as np
import ufl
from typing_extensions import Literal

from .constraints import EqualityConstraint, InequalityConstraint
from .solvers import AugmentedLagrangianMethod, QuadraticPenaltyMethod
from .._exceptions import InputError
from .._optimal_control.optimal_control_problem import OptimalControlProblem
from .._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from ..utils import enlist


class ConstrainedOptimizationProblem(abc.ABC):
    """A PDE constrained optimization problem with additional equality and inequality constraints."""

    def __init__(
        self,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            List[List[fenics.DirichletBC]], List[fenics.DirichletBC], fenics.DirichletBC
        ],
        cost_functional_form: Union[List[ufl.Form], ufl.Form],
        states: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        constraints: Union[
            List[Union[EqualityConstraint, InequalityConstraint]],
            EqualityConstraint,
            InequalityConstraint,
        ],
        config: Optional[configparser.ConfigParser] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[List[List[List[str]]]] = None,
        adjoint_ksp_options: Optional[List[List[List[str]]]] = None,
        scalar_tracking_forms: Optional[List[Dict]] = None,
    ) -> None:
        """
        Parameters
        ----------
        state_forms : ufl.form.Form or list[ufl.form.Form]
            The weak form of the state equation (user implemented). Can be either
            a single UFL form, or a (ordered) list of UFL forms.
        bcs_list : list[fenics.DirichletBC] or list[list[fenics.DirichletBC]] or fenics.DirichletBC or None
            The list of :py:class:`fenics.DirichletBC` objects describing Dirichlet (essential) boundary conditions.
            If this is ``None``, then no Dirichlet boundary conditions are imposed.
        cost_functional_form : ufl.form.Form or list[ufl.form.Form]
            UFL form of the cost functional. Can also be a list of summands of the cost functional
        states : fenics.Function or list[fenics.Function]
            The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
        adjoints : fenics.Function or list[fenics.Function]
            The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
        constraints : EqualityConstraint or InequalityConstraint or list[EqualityConstraint, InequalityConstraint]
            (A list of) additional equality and inequality constraints for the problem.
        config : configparser.ConfigParser or None, optional
            The config file for the problem, generated via :py:func:`cashocs.create_config`.
            Alternatively, this can also be ``None``, in which case the default configurations
            are used, except for the optimization algorithm. This has then to be specified
            in the :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
            default is ``None``.
        initial_guess : list[fenics.Function] or None, optional
            List of functions that act as initial guess for the state variables, should be valid input for :py:func:`fenics.assign`.
            Defaults to ``None``, which means a zero initial guess.
        ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
            A list of strings corresponding to command line options for PETSc,
            used to solve the state systems. If this is ``None``, then the direct solver
            mumps is used (default is ``None``).
        adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
            A list of strings corresponding to command line options for PETSc,
            used to solve the adjoint systems. If this is ``None``, then the same options
            as for the state systems are used (default is ``None``).
        scalar_tracking_forms : dict or list[dict] or None, optional
            A list of dictionaries that define scalar tracking type cost functionals,
            where an integral value should be brought to a desired value. Each dict needs
            to have the keys ``'integrand'`` and ``'tracking_goal'``. Default is ``None``,
            i.e., no scalar tracking terms are considered.
        """

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
        method: Literal[
            "Augmented Lagrangian", "AL", "Quadratic Penalty", "QP"
        ] = "Augmented Lagrangian",
        tol: float = 1e-2,
        max_iter: int = 25,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
        constraint_tol: Optional[float] = None,
        mu_0: Optional[float] = None,
        lambda_0: Optional[List[float]] = None,
    ) -> None:
        """Solves the constrained problem

        Parameters
        ----------
        method : str, optional
            The solution algorithm, either an augmented Lagrangian method ("Augmented Lagrangian",
            "AL") or quadratic penalty method ("Quadratic Penalty", "QP")
        tol : float, optional
            An overall tolerance to be used in the algorithm. This will set the relative
            tolerance for the inner optimization problems to ``tol``. Default is 1e-2.
        max_iter : int, optional
            Maximum number of iterations for the outer solver. Default is 25.
        inner_rtol : float or None, optional
            Relative tolerance for the inner problem. Default is ``None``, so that
            ``inner_rtol = tol`` is used.
        inner_atol : float or None, optional
            Absolute tolerance for the inner problem. Default is ``None``, so that
            ``inner_atol = tol/10`` is used.
        constraint_tol : float or None, optional
            The tolerance for the constraint violation, which is desired. If this is
            ``None`` (default), then this is specified as ``tol/10``.
        mu_0 : float or None, optional
            Initial value of the penalty parameter. Default is ``None``, which means
            that ``mu_0 = 1`` is used.
        lambda_0 : list[float] or None, optional
            Initial guess for the Lagrange multipliers. Default is ``None``, which
            corresponds to a zero guess.

        Returns
        -------
        None
        """

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

    def total_constraint_violation(self) -> float:
        """Compute the total constraint violation.

        Returns
        -------
        float
            The 2-norm of the total constraint violation.
        """

        s = 0.0
        for constraint in self.constraints:
            s += pow(constraint.constraint_violation(), 2)

        return np.sqrt(s)

    @abc.abstractmethod
    def _solve_inner_problem(
        self,
        tol: float = 1e-2,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
    ):
        """Solves the inner (unconstrained) optimization problem.

        Parameters
        ----------
        tol : float, optional
            An overall tolerance to be used in the algorithm. This will set the relative
            tolerance for the inner optimization problems to ``tol``. Default is 1e-2.
        inner_rtol : float or None, optional
            Relative tolerance for the inner problem. Default is ``None``, so that
            ``inner_rtol = tol`` is used.
        inner_atol : float or None, optional
            Absolute tolerance for the inner problem. Default is ``None``, so that
            ``inner_atol = tol/10`` is used.


        Returns
        -------
        None
        """

        pass

    def _pre_hook(self) -> None:
        pass

    def _post_hook(self) -> None:
        pass

    def inject_pre_hook(self, function: Callable) -> None:
        """
        Changes the a-priori hook of the OptimizationProblem

        Parameters
        ----------
        function : function
            A custom function without arguments, which will be called before each solve
            of the state system

        Returns
        -------
        None
        """

        self._pre_hook = function

    def inject_post_hook(self, function: Callable) -> None:
        """
        Changes the a-posteriori hook of the OptimizationProblem

        Parameters
        ----------
        function : function
            A custom function without arguments, which will be called after the computation
            of the gradient(s)

        Returns
        -------
        None
        """

        self._post_hook = function

    def inject_pre_post_hook(
        self, pre_function: Callable, post_function: Callable
    ) -> None:
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
        None
        """

        self.inject_pre_hook(pre_function)
        self.inject_post_hook(post_function)


class ConstrainedOptimalControlProblem(ConstrainedOptimizationProblem):
    def __init__(
        self,
        state_forms: Union[ufl.Form, List[ufl.Form]],
        bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
        cost_functional_form: Union[ufl.Form, List[ufl.Form]],
        states: Union[fenics.Function, List[fenics.Function]],
        controls: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        constraints: Union[
            EqualityConstraint,
            InequalityConstraint,
            List[EqualityConstraint, InequalityConstraint],
        ],
        config: Optional[configparser.ConfigParser] = None,
        riesz_scalar_products: Optional[Union[ufl.Form, List[ufl.Form]]] = None,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[List[List[List[str]]]] = None,
        adjoint_ksp_options: Optional[List[List[List[str]]]] = None,
        scalar_tracking_forms: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        r"""
        Parameters
        ----------
        state_forms : ufl.form.Form or list[ufl.form.Form]
            The weak form of the state equation (user implemented). Can be either
            a single UFL form, or a (ordered) list of UFL forms.
        bcs_list : list[fenics.DirichletBC] or list[list[fenics.DirichletBC]] or fenics.DirichletBC or None
            The list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
            If this is ``None``, then no Dirichlet boundary conditions are imposed.
        cost_functional_form : ufl.form.Form or list[ufl.form.Form]
            UFL form of the cost functional.
        states : fenics.Function or list[fenics.Function]
            The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
        controls : fenics.Function or list[fenics.Function]
            The control variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
        adjoints : fenics.Function or list[fenics.Function]
            The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
        constraints : EqualityConstraint or InequalityConstraint or list[EqualityConstraint, InequalityConstraint]
            (A list of) additional equality and inequality constraints for the problem.
        config : configparser.ConfigParser or None, optional
            The config file for the problem, generated via :py:func:`cashocs.create_config`.
            Alternatively, this can also be ``None``, in which case the default configurations
            are used, except for the optimization algorithm. This has then to be specified
            in the :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
            default is ``None``.
        riesz_scalar_products : None or ufl.form.Form or list[ufl.form.Form], optional
            The scalar products of the control space. Can either be None, a single UFL form, or a
            (ordered) list of UFL forms. If ``None``, the :math:`L^2(\Omega)` product is used.
            (default is ``None``).
        control_constraints : None or list[fenics.Function] or list[float] or list[list[fenics.Function]] or list[list[float]], optional
            Box constraints posed on the control, ``None`` means that there are none (default is ``None``).
            The (inner) lists should contain two elements of the form ``[u_a, u_b]``, where ``u_a`` is the lower,
            and ``u_b`` the upper bound.
        initial_guess : list[fenics.Function], optional
            List of functions that act as initial guess for the state variables, should be valid input for :py:func:`fenics.assign`.
            Defaults to ``None``, which means a zero initial guess.
        ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
            A list of strings corresponding to command line options for PETSc,
            used to solve the state systems. If this is ``None``, then the direct solver
            mumps is used (default is ``None``).
        adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
            A list of strings corresponding to command line options for PETSc,
            used to solve the adjoint systems. If this is ``None``, then the same options
            as for the state systems are used (default is ``None``).
        scalar_tracking_forms : dict or list[dict] or None, optional
            A list of dictionaries that define scalar tracking type cost functionals,
            where an integral value should be brought to a desired value. Each dict needs
            to have the keys ``'integrand'`` and ``'tracking_goal'``. Default is ``None``,
            i.e., no scalar tracking terms are considered.
        """

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

    def _solve_inner_problem(
        self,
        tol: float = 1e-2,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
    ) -> None:
        """Solves the inner (unconstrained) optimization problem.

        Parameters
        ----------
        tol : float, optional
            An overall tolerance to be used in the algorithm. This will set the relative
            tolerance for the inner optimization problems to ``tol``. Default is 1e-2.
        inner_rtol : float or None, optional
            Relative tolerance for the inner problem. Default is ``None``, so that
            ``inner_rtol = tol`` is used.
        inner_atol : float or None, optional
            Absolute tolerance for the inner problem. Default is ``None``, so that
            ``inner_atol = tol/10`` is used.

        Returns
        -------
        None
        """

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
        state_forms: Union[ufl.Form, List[ufl.Form]],
        bcs_list: Union[
            fenics.DirichletBC,
            List[fenics.DirichletBC],
            List[List[fenics.DirichletBC]],
            None,
        ],
        cost_functional_form: Union[ufl.Form, List[ufl.Form]],
        states: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        boundaries: fenics.MeshFunction,
        constraints: Union[
            EqualityConstraint,
            InequalityConstraint,
            List[EqualityConstraint, InequalityConstraint],
        ],
        config: Optional[configparser.ConfigParser] = None,
        shape_scalar_product: Optional[ufl.Form] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[List[List[List[str]]]] = None,
        adjoint_ksp_options: Optional[List[List[List[str]]]] = None,
        scalar_tracking_forms: Optional[Dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        state_forms : ufl.Form or list[ufl.Form]
            The weak form of the state equation (user implemented). Can be either
            a single UFL form, or a (ordered) list of UFL forms.
        bcs_list : list[fenics.DirichletBC] or list[list[fenics.DirichletBC]] or fenics.DirichletBC or None
            The list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
            If this is ``None``, then no Dirichlet boundary conditions are imposed.
        cost_functional_form : ufl.Form or list[ufl.Form]
            UFL form of the cost functional.
        states : fenics.Function or list[fenics.Function]
            The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
        adjoints : fenics.Function or list[fenics.Function]
            The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
        boundaries : fenics.MeshFunction
            :py:class:`fenics.MeshFunction` that indicates the boundary markers.
        constraints : EqualityConstraint or InequalityConstraint or list[EqualityConstraint, InequalityConstraint]
            (A list of) additional equality and inequality constraints for the problem.
        config : configparser.ConfigParser or None, optional
            The config file for the problem, generated via :py:func:`cashocs.create_config`.
            Alternatively, this can also be ``None``, in which case the default configurations
            are used, except for the optimization algorithm. This has then to be specified
            in the :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
            default is ``None``.
        shape_scalar_product : ufl.form.Form or None, optional
            The bilinear form for computing the shape gradient (or gradient deformation).
            This has to use :py:class:`fenics.TrialFunction` and :py:class:`fenics.TestFunction`
            objects to define the weak form, which have to be in a :py:class:`fenics.VectorFunctionSpace`
            of continuous, linear Lagrange finite elements. Moreover, this form is required to be
            symmetric.
        initial_guess : list[fenics.Function] or None, optional
            List of functions that act as initial guess for the state variables, should be valid input for :py:func:`fenics.assign`.
            Defaults to ``None``, which means a zero initial guess.
        ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
            A list of strings corresponding to command line options for PETSc,
            used to solve the state systems. If this is ``None``, then the direct solver
            mumps is used (default is ``None``).
        adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
            A list of strings corresponding to command line options for PETSc,
            used to solve the adjoint systems. If this is ``None``, then the same options
            as for the state systems are used (default is ``None``).
        """

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

    def _solve_inner_problem(
        self,
        tol: float = 1e-2,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
    ) -> None:
        """Solves the inner (unconstrained) optimization problem.

        Parameters
        ----------
        tol : float, optional
            An overall tolerance to be used in the algorithm. This will set the relative
            tolerance for the inner optimization problems to ``tol``. Default is 1e-2.
        inner_rtol : float or None, optional
            Relative tolerance for the inner problem. Default is ``None``, so that
            ``inner_rtol = tol`` is used.
        inner_atol : float or None, optional
            Absolute tolerance for the inner problem. Default is ``None``, so that
            ``inner_atol = tol/10`` is used.

        Returns
        -------
        None
        """

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
