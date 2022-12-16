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

"""Constrained optimization problems."""

from __future__ import annotations

import abc
from typing import Callable, List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
from typing_extensions import Literal
import ufl

from cashocs import _utils
from cashocs._constraints import solvers
from cashocs._optimization import optimal_control
from cashocs._optimization import shape_optimization

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io


class ConstrainedOptimizationProblem(abc.ABC):
    """An optimization problem with additional equality / inequality constraints."""

    solver: Union[solvers.AugmentedLagrangianMethod, solvers.QuadraticPenaltyMethod]

    def __init__(
        self,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            List[List[fenics.DirichletBC]], List[fenics.DirichletBC], fenics.DirichletBC
        ],
        cost_functional_form: Union[
            List[_typing.CostFunctional], _typing.CostFunctional
        ],
        states: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        constraint_list: Union[List[_typing.Constraint], _typing.Constraint],
        config: Optional[io.Config] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[_typing.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[_typing.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
    ) -> None:
        """Initializes self.

        Args:
            state_forms: The weak form of the state equation (user implemented). Can be
                either a single UFL form, or a (ordered) list of UFL forms.
            bcs_list: The list of :py:class:`fenics.DirichletBC` objects describing
                Dirichlet (essential) boundary conditions. If this is ``None``, then no
                Dirichlet boundary conditions are imposed.
            cost_functional_form: UFL form of the cost functional. Can also be a list of
                summands of the cost functional
            states: The state variable(s), can either be a :py:class:`fenics.Function`,
                or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            constraint_list: (A list of) additional equality and inequality constraints
                for the problem.
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).

        """
        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.states = states
        self.adjoints = adjoints

        if config is not None:
            self.config = config
        else:
            self.config = io.Config()
        self.config.validate_config()

        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options

        self.current_function_value = 0.0

        self._pre_callback: Optional[Callable] = None
        self._post_callback: Optional[Callable] = None

        self.cost_functional_form_initial: List[_typing.CostFunctional] = _utils.enlist(
            cost_functional_form
        )
        self.constraint_list: List[_typing.Constraint] = _utils.enlist(constraint_list)

        self.constraint_dim = len(self.constraint_list)

        self.initial_norm = 0.0
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
        """Solves the constrained optimization problem.

        Args:
            method: The solution algorithm, either an augmented Lagrangian method
                ("Augmented Lagrangian", "AL") or quadratic penalty method
                ("Quadratic Penalty", "QP")
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            max_iter: Maximum number of iterations for the outer solver. Default is 25.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
            so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            constraint_tol: The tolerance for the constraint violation, which is
                desired. If this is ``None`` (default), then this is specified as
                ``tol/10``.
            mu_0: Initial value of the penalty parameter. Default is ``None``, which
                means that ``mu_0 = 1`` is used.
            lambda_0: Initial guess for the Lagrange multipliers. Default is ``None``,
                which corresponds to a zero guess.

        """
        if method.casefold() in ["augmented lagrangian", "al"]:
            self.solver = solvers.AugmentedLagrangianMethod(
                self, mu_0=mu_0, lambda_0=lambda_0
            )
        elif method.casefold() in ["quadratic penalty", "qp"]:
            self.solver = solvers.QuadraticPenaltyMethod(
                self, mu_0=mu_0, lambda_0=lambda_0
            )

        self.solver.solve(
            tol=tol,
            max_iter=max_iter,
            inner_rtol=inner_rtol,
            inner_atol=inner_atol,
            constraint_tol=constraint_tol,
        )

    def total_constraint_violation(self) -> float:
        """Computes the total constraint violation.

        Returns:
            The 2-norm of the total constraint violation.

        """
        s = 0.0
        for constraint in self.constraint_list:
            s += pow(constraint.constraint_violation(), 2)

        violation: float = np.sqrt(s)
        return violation

    @abc.abstractmethod
    def _solve_inner_problem(
        self,
        tol: float = 1e-2,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
    ) -> None:
        """Solves the inner (unconstrained) optimization problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.

        """
        self.rtol = inner_rtol or tol

    def inject_pre_callback(self, function: Optional[Callable]) -> None:
        """Changes the a-priori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self._pre_callback = function

    def inject_post_callback(self, function: Optional[Callable]) -> None:
        """Changes the a-posteriori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self._post_callback = function

    def inject_pre_post_callback(
        self, pre_function: Optional[Callable], post_function: Optional[Callable]
    ) -> None:
        """Changes the a-priori (pre) and a-posteriori (post) callbacks of the problem.

        Args:
            pre_function: A function without arguments, which is to be called before
                each solve of the state system
            post_function: A function without arguments, which is to be called after
                each computation of the (shape) gradient

        """
        self.inject_pre_callback(pre_function)
        self.inject_post_callback(post_function)


class ConstrainedOptimalControlProblem(ConstrainedOptimizationProblem):
    """An optimal control problem with additional (in-)equality constraints."""

    def __init__(
        self,
        state_forms: Union[ufl.Form, List[ufl.Form]],
        bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
        cost_functional_form: Union[
            List[_typing.CostFunctional], _typing.CostFunctional
        ],
        states: Union[fenics.Function, List[fenics.Function]],
        controls: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        constraint_list: Union[_typing.Constraint, List[_typing.Constraint]],
        config: Optional[io.Config] = None,
        riesz_scalar_products: Optional[Union[ufl.Form, List[ufl.Form]]] = None,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[_typing.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[_typing.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        control_bcs_list: Optional[
            Union[
                fenics.DirichletBC,
                List[fenics.DirichletBC],
                List[List[fenics.DirichletBC]],
            ]
        ] = None,
    ) -> None:
        r"""Initializes self.

        Args:
            state_forms: The weak form of the state equation (user implemented). Can be
                either a single UFL form, or a (ordered) list of UFL forms.
            bcs_list: The list of :py:class:`fenics.DirichletBC` objects describing
                Dirichlet (essential) boundary conditions. If this is ``None``, then no
                Dirichlet boundary conditions are imposed.
            cost_functional_form: UFL form of the cost functional. Can also be a list of
                summands of the cost functional
            states: The state variable(s), can either be a :py:class:`fenics.Function`,
                or a list of these.
            controls: The control variable(s), can either be a
                :py:class:`fenics.Function`, or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            constraint_list: (A list of) additional equality and inequality constraints
                for the problem.
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            riesz_scalar_products: The scalar products of the control space. Can either
                be None, a single UFL form, or a (ordered) list of UFL forms. If
                ``None``, the :math:`L^2(\Omega)` product is used (default is ``None``).
            control_constraints: Box constraints posed on the control, ``None`` means
                that there are none (default is ``None``). The (inner) lists should
                contain two elements of the form ``[u_a, u_b]``, where ``u_a`` is the
                lower, and ``u_b`` the upper bound.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).
            control_bcs_list: A list of boundary conditions for the control variables.
                This is passed analogously to ``bcs_list``. Default is ``None``.

        """
        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            constraint_list,
            config,
            initial_guess=initial_guess,
            ksp_options=ksp_options,
            adjoint_ksp_options=adjoint_ksp_options,
        )

        self.controls = controls
        self.riesz_scalar_products = riesz_scalar_products
        self.control_bcs_list = control_bcs_list
        self.control_constraints = control_constraints

    def _solve_inner_problem(
        self,
        tol: float = 1e-2,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
    ) -> None:
        """Solves the inner (unconstrained) optimization problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.

        """
        super()._solve_inner_problem(tol, inner_rtol, inner_atol)

        optimal_control_problem = optimal_control.OptimalControlProblem(
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
            control_bcs_list=self.control_bcs_list,
        )

        optimal_control_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )
        optimal_control_problem.shift_cost_functional(
            self.solver.inner_cost_functional_shift
        )

        if inner_atol is not None:
            atol = inner_atol
        else:
            atol = self.initial_norm * tol / 10.0

        optimal_control_problem.solve(rtol=self.rtol, atol=atol)
        if self.solver.iterations == 1:
            self.initial_norm = optimal_control_problem.solver.gradient_norm_initial

        temp_problem = optimal_control.OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form_initial,
            self.states,
            self.controls,
            self.adjoints,
            config=self.config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
        )
        temp_problem.state_problem.has_solution = True
        self.current_function_value = temp_problem.reduced_cost_functional.evaluate()


class ConstrainedShapeOptimizationProblem(ConstrainedOptimizationProblem):
    """A shape optimization problem with additional (in-)equality constraints."""

    def __init__(
        self,
        state_forms: Union[ufl.Form, List[ufl.Form]],
        bcs_list: Union[
            fenics.DirichletBC,
            List[fenics.DirichletBC],
            List[List[fenics.DirichletBC]],
            None,
        ],
        cost_functional_form: Union[
            List[_typing.CostFunctional], _typing.CostFunctional
        ],
        states: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        boundaries: fenics.MeshFunction,
        constraint_list: Union[_typing.Constraint, List[_typing.Constraint]],
        config: Optional[io.Config] = None,
        shape_scalar_product: Optional[ufl.Form] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[_typing.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[_typing.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
    ) -> None:
        """Initializes self.

        Args:
            state_forms: The weak form of the state equation (user implemented). Can be
                either a single UFL form, or a (ordered) list of UFL forms.
            bcs_list: The list of :py:class:`fenics.DirichletBC` objects describing
                Dirichlet (essential) boundary conditions. If this is ``None``, then no
                Dirichlet boundary conditions are imposed.
            cost_functional_form: UFL form of the cost functional. Can also be a list of
                summands of the cost functional
            states: The state variable(s), can either be a :py:class:`fenics.Function`,
                or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            boundaries: A :py:class:`fenics.MeshFunction` that indicates the boundary
                markers.
            constraint_list: (A list of) additional equality and inequality constraints
                for the problem.
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            shape_scalar_product: The bilinear form for computing the shape gradient
                (or gradient deformation). This has to use
                :py:class:`fenics.TrialFunction` and :py:class:`fenics.TestFunction`
                objects to define the weak form, which have to be in a
                :py:class:`fenics.VectorFunctionSpace` of continuous, linear Lagrange
                finite elements. Moreover, this form is required to be symmetric.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).

        """
        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            constraint_list,
            config=config,
            initial_guess=initial_guess,
            ksp_options=ksp_options,
            adjoint_ksp_options=adjoint_ksp_options,
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

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.

        """
        super()._solve_inner_problem(tol, inner_rtol, inner_atol)

        shape_optimization_problem = shape_optimization.ShapeOptimizationProblem(
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
        )
        shape_optimization_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )
        shape_optimization_problem.shift_cost_functional(
            self.solver.inner_cost_functional_shift
        )

        if inner_atol is not None:
            atol = inner_atol
        else:
            atol = self.initial_norm * tol / 10.0

        shape_optimization_problem.solve(rtol=self.rtol, atol=atol)
        if self.solver.iterations == 1:
            self.initial_norm = shape_optimization_problem.solver.gradient_norm_initial

        temp_problem = shape_optimization.ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form_initial,
            self.states,
            self.adjoints,
            self.boundaries,
            config=self.config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
        )
        temp_problem.state_problem.has_solution = True
        self.current_function_value = temp_problem.reduced_cost_functional.evaluate()
