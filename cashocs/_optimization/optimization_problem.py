# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

"""PDE constrained optimization problems.

This module is used to define the parent class for the optimization problems,
as many parameters and variables are common for optimal control and shape
optimization problems.
"""

from __future__ import annotations

import abc
import copy
from typing import Callable, cast, TYPE_CHECKING
import weakref

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _forms
from cashocs import _pde_problems
from cashocs import _utils
from cashocs import io
from cashocs import log
from cashocs._database import database
from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    from cashocs_extensions import mesh_quality_constraints

    from cashocs import _typing
    from cashocs._optimization import line_search as ls
    from cashocs._optimization import optimization_algorithms
    from cashocs._optimization import optimization_variable_abstractions as ova
    from cashocs._optimization.topology_optimization import (
        topology_optimization_algorithm,
    )


class OptimizationProblem(abc.ABC):
    """Base class for an abstract PDE constrained optimization problem.

    This class performs the initialization of the shared input so that the rest
    of cashocs can use it directly. Additionally, it includes methods that
    can be used to compute the state and adjoint variables by solving the
    corresponding equations. This could be subclassed to generate custom
    optimization problems.
    """

    gradient: list[fenics.Function]
    reduced_cost_functional: cost_functional.ReducedCostFunctional
    gradient_problem: _typing.GradientProblem
    output_manager: io.OutputManager
    form_handler: _typing.FormHandler
    optimization_variable_abstractions: ova.OptimizationVariableAbstractions
    adjoint_problem: _pde_problems.AdjointProblem
    state_problem: _pde_problems.StateProblem
    uses_custom_scalar_product: bool = False
    line_search: ls.LineSearch
    hessian_problem: _pde_problems.HessianProblem
    solver: (
        optimization_algorithms.OptimizationAlgorithm
        | topology_optimization_algorithm.TopologyOptimizationAlgorithm
    )
    config: io.Config
    initial_guess: list[fenics.Function] | None
    cost_functional_list: list[_typing.CostFunctional]

    def __init__(
        self,
        state_forms: list[ufl.Form] | ufl.Form,
        bcs_list: (
            list[list[fenics.DirichletBC]]
            | list[fenics.DirichletBC]
            | fenics.DirichletBC
        ),
        cost_functional_form: list[_typing.CostFunctional] | _typing.CostFunctional,
        states: list[fenics.Function] | fenics.Function,
        adjoints: list[fenics.Function] | fenics.Function,
        config: io.Config | None = None,
        initial_guess: list[fenics.Function] | fenics.Function | None = None,
        ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        adjoint_ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        gradient_ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        desired_weights: list[float] | None = None,
        temp_dict: dict | None = None,
        initial_function_values: list[float] | None = None,
        preconditioner_forms: list[ufl.Form] | ufl.Form | None = None,
        pre_callback: (
            Callable[[], None] | Callable[[_typing.OptimizationProblem], None] | None
        ) = None,
        post_callback: (
            Callable[[], None] | Callable[[_typing.OptimizationProblem], None] | None
        ) = None,
        linear_solver: _utils.linalg.LinearSolver | None = None,
        adjoint_linear_solver: _utils.linalg.LinearSolver | None = None,
        newton_linearizations: ufl.Form | list[ufl.Form] | None = None,
        excluded_from_time_derivative: (
            list[int] | list[list[int]] | list[None] | None
        ) = None,
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
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            initial_guess: list of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of dicts corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of dicts corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).
            gradient_ksp_options: A list of dicts corresponding to command line options
                for PETSc, used to compute the (shape) gradient. If this is ``None``,
                either a direct or an iterative method is used (depending on the
                configuration, section OptimizationRoutine, key gradient_method).
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.
            temp_dict: This is a private parameter of the class, required for remeshing.
                This parameter must not be set by the user and should be ignored.
                Using this parameter may result in unintended side effects.
            initial_function_values: This is a privatve parameter of the class, required
                for remeshing. This parameter must not be set by the user and should be
                ignored. Using this parameter may result in unintended side effects.
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.
            pre_callback: A function (without arguments) that will be called before each
                solve of the state system
            post_callback: A function (without arguments) that will be called after the
                computation of the gradient.
            linear_solver: The linear solver (KSP) which is used to solve the linear
                systems arising from the discretized PDE.
            adjoint_linear_solver: The linear solver (KSP) which is used to solve the
                (linear) adjoint system.
            newton_linearizations: A (list of) UFL forms describing which (alternative)
                linearizations should be used for the (nonlinear) state equations when
                solving them (with Newton's method). The default is `None`, so that the
                Jacobian of the supplied state forms is used.
            excluded_from_time_derivative: For each state equation, a list of indices
                which are not part of the first order time derivative for pseudo time
                stepping. Example: Pressure for incompressible flow. Default is None.

        Notes:
            If one uses a single PDE constraint, the inputs can be the objects
            (UFL forms, functions, etc.) directly. In case multiple PDE constraints
            are present the inputs have to be put into (ordered) lists. The order of
            the objects depends on the order of the state variables, so that
            ``state_forms[i]`` is the weak form of the PDE for ``states[i]`` with
            boundary conditions ``bcs_list[i]`` and corresponding adjoint state
            ``adjoints[i]``.

        """
        self.state_forms = _utils.enlist(state_forms)
        self.state_dim = len(self.state_forms)
        self.bcs_list = _utils.check_and_enlist_bcs(bcs_list)

        self.cost_functional_list = self._parse_cost_functional_form(
            cost_functional_form
        )

        self.states: list[fenics.Function] = _utils.enlist(states)
        self.adjoints: list[fenics.Function] = _utils.enlist(adjoints)

        self.use_scaling = False

        if config is None:
            self.config = io.Config()
        else:
            self.config = copy.deepcopy(config)
        self.config.validate_config()

        (
            self.initial_guess,
            self.ksp_options,
            self.adjoint_ksp_options,
            self.gradient_ksp_options,
            self.desired_weights,
            self.preconditioner_forms,
            self.newton_linearizations,
            self.excluded_from_time_derivative,
        ) = self._parse_optional_inputs(
            initial_guess,
            ksp_options,
            adjoint_ksp_options,
            gradient_ksp_options,
            desired_weights,
            preconditioner_forms,
            newton_linearizations,
            excluded_from_time_derivative,
        )

        if self.config.getboolean("StateSystem", "use_adjoint_linearizations"):
            self.adjoint_linearizations: list | None = []
            for derivative in self.newton_linearizations:
                if derivative is not None:
                    args = derivative.arguments()
                    adjoint_linearization = ufl.replace(
                        derivative, {args[0]: args[1], args[1]: args[0]}
                    )
                    self.adjoint_linearizations.append(adjoint_linearization)
                else:
                    self.adjoint_linearizations.append(None)
        else:
            self.adjoint_linearizations = None

        if initial_function_values is not None:
            self.initial_function_values = initial_function_values

        self.algorithm = _utils.optimization_algorithm_configuration(self.config)

        self.has_custom_adjoint = False
        self.has_custom_derivative = False

        self.db = database.Database(
            self.config,
            self.states,
            self.adjoints,
            self.ksp_options,
            self.adjoint_ksp_options,
            self.gradient_ksp_options,
            self.cost_functional_list,
            self.state_forms,
            self.bcs_list,
            self.preconditioner_forms,
        )

        self.linear_solver = linear_solver
        self.adjoint_linear_solver = adjoint_linear_solver

        self.db.callback.pre_callback = pre_callback
        self.db.callback.post_callback = post_callback
        self.db.callback.problem = weakref.proxy(self)

        if temp_dict is not None:
            self.db.parameter_db.temp_dict.update(temp_dict)
            self.db.parameter_db.is_remeshed = True

        self.general_form_handler = _forms.GeneralFormHandler(self.db)
        self.state_problem = _pde_problems.StateProblem(
            self.db,
            self.general_form_handler.state_form_handler,
            self.initial_guess,
            linear_solver=self.linear_solver,
            newton_linearizations=self.newton_linearizations,
            excluded_from_time_derivative=self.excluded_from_time_derivative,
        )
        self.adjoint_problem = _pde_problems.AdjointProblem(
            self.db,
            self.general_form_handler.adjoint_form_handler,
            self.state_problem,
            linear_solver=self.adjoint_linear_solver,
            adjoint_linearizations=self.adjoint_linearizations,
        )
        self.constraint_manager: mesh_quality_constraints.ConstraintManager | None = (
            None
        )

        self.output_manager = io.OutputManager(self.db)

    @abc.abstractmethod
    def _erase_pde_memory(self) -> None:
        """Ensures that the PDEs are solved again, and no cache is used."""
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False

    def _parse_cost_functional_form(
        self,
        cost_functional_form: (
            list[_typing.CostFunctional]
            | _typing.CostFunctional
            | list[ufl.Form]
            | ufl.Form
        ),
    ) -> list[_typing.CostFunctional]:
        """Parses the cost functional form for use in cashocs.

        Returns:
            The list of cost functionals.

        """
        input_cost_functional_list = _utils.enlist(cost_functional_form)
        cost_functional_list: list[_typing.CostFunctional] = []
        for functional in input_cost_functional_list:
            if isinstance(
                functional,
                (
                    cost_functional.IntegralFunctional,
                    cost_functional.ScalarTrackingFunctional,
                    cost_functional.MinMaxFunctional,
                    cost_functional.Functional,
                ),
            ):
                cost_functional_list.append(functional)
            else:
                raise _exceptions.InputError(
                    "cashocs.OptimizationProblem",
                    "cost_functional_list",
                    "You have supplied a wrong cost functional.\n"
                    "Starting with cashocs v2.0 only a list of "
                    "cashocs.IntegralFunctional, cashocs.ScalarTrackingFunctional, "
                    "or cashocs.MinMaxFunctional is allowed.",
                )

        return cost_functional_list

    def _parse_optional_inputs(
        self,
        initial_guess: list[fenics.Function] | fenics.Function | None,
        ksp_options: _typing.KspOption | list[_typing.KspOption] | None,
        adjoint_ksp_options: _typing.KspOption | list[_typing.KspOption] | None,
        gradient_ksp_options: _typing.KspOption | list[_typing.KspOption] | None,
        desired_weights: list[float] | float | None,
        preconditioner_forms: list[ufl.Form] | ufl.Form | None,
        newton_linearizations: ufl.Form | list[ufl.Form] | None = None,
        excluded_from_time_derivative: (
            list[int] | list[list[int]] | list[None] | None
        ) = None,
    ) -> tuple[
        list[fenics.Function] | None,
        list[_typing.KspOption],
        list[_typing.KspOption],
        list[_typing.KspOption] | None,
        list[float] | None,
        list[ufl.Form | None],
        list[ufl.Form | None],
        list[list[int]] | list[None],
    ]:
        """Initializes the optional input parameters.

        Args:
            initial_guess: list of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`.
            ksp_options: A list of dicts corresponding to command line options for
                PETSc, used to solve the state systems.
            adjoint_ksp_options: A list of dicts corresponding to command line options
                for PETSc, used to solve the adjoint systems.
            gradient_ksp_options: A list of dicts corresponding to command line options
                for computing the gradient with PETSc.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration.
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.
            newton_linearizations: A (list of) UFL forms describing which (alternative)
                linearizations should be used for the (nonlinear) state equations when
                solving them (with Newton's method). The default is `None`, so that the
                Jacobian of the supplied state forms is used.
            excluded_from_time_derivative: For each state equation, a list of indices
                which are not part of the first order time derivative for pseudo time
                stepping. Example: Pressure for incompressible flow. Default is None.

        """
        if initial_guess is None:
            parsed_initial_guess = initial_guess
        else:
            parsed_initial_guess = _utils.enlist(initial_guess)

        if ksp_options is None:
            parsed_ksp_options: list[_typing.KspOption] = []
            option = copy.deepcopy(_utils.linalg.direct_ksp_options)

            for _ in range(self.state_dim):
                parsed_ksp_options.append(option)
        else:
            parsed_ksp_options = _utils.enlist(ksp_options)

        parsed_adjoint_ksp_options: list[_typing.KspOption] = (
            copy.deepcopy(parsed_ksp_options)
            if adjoint_ksp_options is None
            else _utils.enlist(adjoint_ksp_options)
        )

        if gradient_ksp_options is None:
            parsed_gradient_ksp_options = None
        else:
            parsed_gradient_ksp_options = _utils.enlist(gradient_ksp_options)

        if desired_weights is None:
            parsed_desired_weights = desired_weights
        else:
            parsed_desired_weights = _utils.enlist(desired_weights)
            self.use_scaling = True

        if preconditioner_forms is None:
            parsed_preconditioner_forms: list[ufl.Form | None] = []
            for _ in range(self.state_dim):
                parsed_preconditioner_forms.append(None)
        else:
            parsed_preconditioner_forms = _utils.enlist(preconditioner_forms)

        if newton_linearizations is None:
            parsed_newton_linearizations: list[ufl.Form | None] = []
            for _ in range(self.state_dim):
                parsed_newton_linearizations.append(None)
        else:
            parsed_newton_linearizations = _utils.enlist(newton_linearizations)

        parsed_time_derivative_list = self._parse_excluded_from_time_derivative(
            excluded_from_time_derivative
        )

        return (
            parsed_initial_guess,
            parsed_ksp_options,
            parsed_adjoint_ksp_options,
            parsed_gradient_ksp_options,
            parsed_desired_weights,
            parsed_preconditioner_forms,
            parsed_newton_linearizations,
            parsed_time_derivative_list,
        )

    def _parse_excluded_from_time_derivative(
        self,
        excluded_from_time_derivative: (
            list[int] | list[list[int]] | list[None] | None
        ) = None,
    ) -> list[list[int]] | list[None]:
        """Initiale the excluded_from_time_derivative parameter.

        Args:
            excluded_from_time_derivative: For each state equation, a list of indices
                which are not part of the first order time derivative for pseudo time
                stepping. Example: Pressure for incompressible flow. Default is None.

        Returns:
            The parsed parameter.

        """
        if excluded_from_time_derivative is None:
            parsed_time_derivative_list: list[list[int]] | list[None] = [
                None
            ] * self.state_dim
        else:
            if len(excluded_from_time_derivative) > 0 and isinstance(
                excluded_from_time_derivative[0], int
            ):
                excluded_from_time_derivative = cast(
                    list[int], excluded_from_time_derivative
                )
                parsed_time_derivative_list = [excluded_from_time_derivative]
            else:
                excluded_from_time_derivative = cast(
                    list[list[int]], excluded_from_time_derivative
                )
                parsed_time_derivative_list = excluded_from_time_derivative

        return parsed_time_derivative_list

    def compute_state_variables(self) -> None:
        """Solves the state system.

        This can be used for debugging purposes and to validate the solver.
        Updates and overwrites the user input for the state variables.
        """
        self.state_problem.solve()

    def compute_adjoint_variables(self) -> None:
        """Solves the adjoint system.

        This can be used for debugging purposes and solver validation.
        Updates / overwrites the user input for the adjoint variables.
        The solution of the corresponding state system needed to determine
        the adjoints is carried out automatically.
        """
        self.state_problem.solve()
        self.adjoint_problem.solve()

    def supply_adjoint_forms(
        self,
        adjoint_forms: ufl.Form | list[ufl.Form],
        adjoint_bcs_list: (
            fenics.DirichletBC
            | list[fenics.DirichletBC]
            | list[list[fenics.DirichletBC]]
        ),
    ) -> None:
        """Overwrites the computed weak forms of the adjoint system.

        This allows the user to specify their own weak forms of the problems and to use
        cashocs merely as a solver for solving the optimization problems.

        Args:
            adjoint_forms: The UFL forms of the adjoint system(s).
            adjoint_bcs_list: The list of Dirichlet boundary conditions for the adjoint
                system(s).

        """
        mod_forms = _utils.enlist(adjoint_forms)
        mod_bcs_list: list
        if adjoint_bcs_list == [] or adjoint_bcs_list is None:
            mod_bcs_list = []
            for _ in range(self.state_dim):
                mod_bcs_list.append([])
        else:
            mod_bcs_list = _utils.check_and_enlist_bcs(adjoint_bcs_list)

        self.general_form_handler.adjoint_form_handler.bcs_list_ad = mod_bcs_list

        self.general_form_handler.adjoint_form_handler.adjoint_eq_forms = mod_forms
        # replace the adjoint function by a TrialFunction for internal use
        repl_forms = [
            ufl.replace(
                mod_forms[i],
                {self.adjoints[i]: self.db.function_db.trial_functions_adjoint[i]},
            )
            for i in range(self.state_dim)
        ]
        self.general_form_handler.adjoint_form_handler.linear_adjoint_eq_forms = (
            repl_forms
        )

        (
            self.general_form_handler.adjoint_form_handler.adjoint_eq_lhs,
            self.general_form_handler.adjoint_form_handler.adjoint_eq_rhs,
        ) = _utils.split_linear_forms(
            self.general_form_handler.adjoint_form_handler.linear_adjoint_eq_forms
        )

        self.has_custom_adjoint = True

    def _check_for_custom_forms(self) -> None:
        """Checks, whether custom user forms are used and if they are compatible."""
        if self.has_custom_adjoint and not self.has_custom_derivative:
            log.warning(
                "You only supplied the adjoint system. "
                "This might lead to unexpected results.\n"
                "Consider also supplying the (shape) derivative "
                "of the reduced cost functional,"
                "or check your approach with the cashocs.verification module."
            )

        elif not self.has_custom_adjoint and self.has_custom_derivative:
            log.warning(
                "You only supplied the derivative of the reduced cost functional. "
                "This might lead to unexpected results.\n"
                "Consider also supplying the adjoint system, "
                "or check your approach with the cashocs.verification module."
            )

        if self.algorithm.casefold() == "newton" and (
            self.has_custom_adjoint or self.has_custom_derivative
        ):
            raise _exceptions.InputError(
                "cashocs.optimization_problem.OptimizationProblem",
                "solve",
                "The usage of custom forms is not compatible with the Newton solver."
                "Please do not supply custom forms if you want to use "
                "the Newton solver.",
            )

    def inject_pre_callback(self, function: Callable | None) -> None:
        """Changes the a-priori callback of the OptimizationProblem to function.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self.db.callback.pre_callback = function
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False

    def inject_post_callback(self, function: Callable | None) -> None:
        """Changes the a-posteriori callback of the OptimizationProblem to function.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self.db.callback.post_callback = function
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False

    def inject_pre_post_callback(
        self, pre_function: Callable | None, post_function: Callable | None
    ) -> None:
        """Changes the a-priori (pre) and a-posteriori (post) callbacks.

        Args:
            pre_function: A function without arguments, which is to be called before
                each solve of the state system
            post_function: A function without arguments, which is to be called after
                each computation of the (shape) gradient

        """
        self.inject_pre_callback(pre_function)
        self.inject_post_callback(post_function)

    def initialize_solve_parameters(
        self,
        algorithm: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        r"""Solves the optimization problem by the method specified in the config file.

        Args:
            algorithm: Selects the optimization algorithm. Valid choices are
                ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
                ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
                for nonlinear conjugate gradient methods, and ``'lbfgs'`` or ``'bfgs'``
                for limited memory BFGS methods. This overwrites the value specified
                in the config file. If this is ``None``, then the value in the
                config file is used. Default is ``None``. In addition, for optimal
                control problems, one can use ``'newton'`` for a truncated Newton
                method.
            rtol: The relative tolerance used for the termination criterion. Overwrites
                the value specified in the config file. If this is ``None``, the value
                from the config file is taken. Default is ``None``.
            atol: The absolute tolerance used for the termination criterion. Overwrites
                the value specified in the config file. If this is ``None``, the value
                from the config file is taken. Default is ``None``.
            max_iter: The maximum number of iterations the optimization algorithm can
                carry out before it is terminated. Overwrites the value specified in the
                config file. If this is ``None``, the value from the config file is
                taken. Default is ``None``.

        Notes:
            If either ``rtol`` or ``atol`` are specified as arguments to the ``.solve``
            call, the termination criterion changes to:

            - a purely relative one (if only ``rtol`` is specified), i.e.,

            .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

            - a purely absolute one (if only ``atol`` is specified), i.e.,

            .. math:: || \nabla J(u_K) || \leq \texttt{atol}.

            - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

            .. math::

                || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol}
                || \nabla J(u_0) ||

        """
        self.algorithm = _utils.optimization_algorithm_configuration(
            self.config, algorithm
        )

        if (rtol is not None) and (atol is None):
            self.config.set("OptimizationRoutine", "rtol", str(rtol))
            self.config.set("OptimizationRoutine", "atol", str(0.0))
        elif (atol is not None) and (rtol is None):
            self.config.set("OptimizationRoutine", "rtol", str(0.0))
            self.config.set("OptimizationRoutine", "atol", str(atol))
        elif (atol is not None) and (rtol is not None):
            self.config.set("OptimizationRoutine", "rtol", str(rtol))
            self.config.set("OptimizationRoutine", "atol", str(atol))

        if max_iter is not None:
            self.config.set("OptimizationRoutine", "max_iter", str(max_iter))

    @abc.abstractmethod
    def solve(
        self,
        algorithm: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        r"""Solves the optimization problem by the method specified in the config file.

        Args:
            algorithm: Selects the optimization algorithm. Valid choices are
                ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
                ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
                for nonlinear conjugate gradient methods, and ``'lbfgs'`` or ``'bfgs'``
                for limited memory BFGS methods. This overwrites the value specified
                in the config file. If this is ``None``, then the value in the
                config file is used. Default is ``None``. In addition, for optimal
                control problems, one can use ``'newton'`` for a truncated Newton
                method.
            rtol: The relative tolerance used for the termination criterion. Overwrites
                the value specified in the config file. If this is ``None``, the value
                from the config file is taken. Default is ``None``.
            atol: The absolute tolerance used for the termination criterion. Overwrites
                the value specified in the config file. If this is ``None``, the value
                from the config file is taken. Default is ``None``.
            max_iter: The maximum number of iterations the optimization algorithm can
                carry out before it is terminated. Overwrites the value specified in the
                config file. If this is ``None``, the value from the config file is
                taken. Default is ``None``.

        Notes:
            If either ``rtol`` or ``atol`` are specified as arguments to the ``.solve``
            call, the termination criterion changes to:

            - a purely relative one (if only ``rtol`` is specified), i.e.,

            .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

            - a purely absolute one (if only ``atol`` is specified), i.e.,

            .. math:: || \nabla J(u_K) || \leq \texttt{atol}.

            - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

            .. math::

                || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol}
                || \nabla J(u_0) ||

        """
        self.initialize_solve_parameters(algorithm, rtol, atol, max_iter)
        self.config.validate_config()
        self._check_for_custom_forms()

    def shift_cost_functional(self, shift: float = 0.0) -> None:
        """Shifts the cost functional by a constant.

        Args:
            shift: The constant, by which the cost functional is shifted.

        """
        self.form_handler.cost_functional_shift = shift

    @abc.abstractmethod
    def gradient_test(self) -> float:
        """Test the correctness of the computed gradient with finite differences.

        Returns:
            The result of the gradient test. If this is (approximately) 2 or larger,
            everything works as expected.

        """
        pass

    def _compute_initial_function_values(self) -> None:
        """Computes the cost functional values for the initial iteration."""
        self.state_problem.solve()
        self.initial_function_values = []
        for i, functional in enumerate(self.cost_functional_list):
            val = functional.evaluate()

            if abs(val) <= 1e-15:
                val = 1.0
                log.info(
                    f"Term {i:d} of the cost functional vanishes "
                    f"for the initial iteration. Multiplying this term with the "
                    f"factor you supplied in desired weights."
                )

            self.initial_function_values.append(val)

    def _scale_cost_functional(self) -> None:
        """Scales the terms of the cost functional."""
        log.warning(
            "You are using the automatic scaling functionality of cashocs."
            "This may lead to unexpected results if you try to scale the cost "
            "functional yourself or if you supply custom forms."
        )

        if self.use_scaling and self.desired_weights is not None:
            if not self.db.parameter_db.is_remeshed:
                self._compute_initial_function_values()

            for i, functional in enumerate(self.cost_functional_list):
                scaling_factor = np.abs(
                    self.desired_weights[i] / self.initial_function_values[i]
                )
                functional.scale(scaling_factor)
