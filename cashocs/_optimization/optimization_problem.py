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

"""Base classes for the PDE constrained optimization problems.

This module is used to define the parent class for the optimization problems,
as many parameters and variables are common for optimal control and shape
optimization problems.
"""

from __future__ import annotations

import abc
import copy
import json
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
import ufl

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils
from cashocs import io
from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    from cashocs import _pde_problems
    from cashocs import types
    from cashocs._optimization import line_search as ls
    from cashocs._optimization import optimization_algorithms
    from cashocs._optimization import optimization_variable_abstractions as ova


class OptimizationProblem(abc.ABC):
    """Base class for an abstract PDE constrained optimization problem.

    This class performs the initialization of the shared input so that the rest
    of cashocs can use it directly. Additionally, it includes methods that
    can be used to compute the state and adjoint variables by solving the
    corresponding equations. This could be subclassed to generate custom
    optimization problems.
    """

    gradient: List[fenics.Function]
    reduced_cost_functional: cost_functional.ReducedCostFunctional
    gradient_problem: types.GradientProblem
    output_manager: io.OutputManager
    form_handler: types.FormHandler
    optimization_variable_abstractions: ova.OptimizationVariableAbstractions
    adjoint_problem: _pde_problems.AdjointProblem
    state_problem: _pde_problems.StateProblem
    uses_custom_scalar_product: bool = False
    use_scalar_tracking: bool = False
    temp_dict: Optional[Dict]
    algorithm: str
    line_search: ls.LineSearch
    hessian_problem: _pde_problems.HessianProblem
    solver: optimization_algorithms.OptimizationAlgorithm
    config: io.Config
    initial_guess: Optional[List[fenics.Function]]
    cost_functional_list: List[types.CostFunctional]

    def __init__(
        self,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            List[List[fenics.DirichletBC]], List[fenics.DirichletBC], fenics.DirichletBC
        ],
        cost_functional_form: Union[
            List[types.CostFunctional], types.CostFunctional, List[ufl.Form], ufl.Form
        ],
        states: Union[List[fenics.Function], fenics.Function],
        adjoints: Union[List[fenics.Function], fenics.Function],
        config: Optional[io.Config] = None,
        initial_guess: Optional[Union[List[fenics.Function], fenics.Function]] = None,
        ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
        min_max_terms: Optional[Union[List[Dict], Dict]] = None,
        desired_weights: Optional[List[float]] = None,
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
                :py:func:`cashocs.create_config`. Alternatively, this can also be
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
            scalar_tracking_forms: A list of dictionaries that define scalar tracking
                type cost functionals, where an integral value should be brought to a
                desired value. Each dict needs to have the keys ``'integrand'`` and
                ``'tracking_goal'``. Default is ``None``, i.e., no scalar tracking terms
                are considered.
            min_max_terms: Additional terms for the cost functional, not be used
                directly.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.

        Notes:
            If one uses a single PDE constraint, the inputs can be the objects
            (UFL forms, functions, etc.) directly. In case multiple PDE constraints
            are present the inputs have to be put into (ordered) lists. The order of
            the objects depends on the order of the state variables, so that
            ``state_forms[i]`` is the weak form of the PDE for ``states[i]`` with
            boundary conditions ``bcs_list[i]`` and corresponding adjoint state
            ``adjoints[i]``.

        """
        self.has_cashocs_remesh_flag, self.temp_dir = _utils.parse_remesh()

        self.state_forms = _utils.enlist(state_forms)
        self.state_dim = len(self.state_forms)
        self.bcs_list = _utils.check_and_enlist_bcs(bcs_list)

        self._parse_cost_functional_form(cost_functional_form)

        self.states: List[fenics.Function] = _utils.enlist(states)
        self.adjoints: List[fenics.Function] = _utils.enlist(adjoints)

        self.use_min_max_terms = False
        self.use_scaling = False

        self._parse_optional_inputs(
            config,
            initial_guess,
            ksp_options,
            adjoint_ksp_options,
            scalar_tracking_forms,
            min_max_terms,
            desired_weights,
        )

        fenics.set_log_level(fenics.LogLevel.CRITICAL)

        self.has_custom_adjoint = False
        self.has_custom_derivative = False

        self._is_shape_problem = False
        self._is_control_problem = False

    @property
    def is_shape_problem(self) -> bool:
        """Returns a boolean flag, ``True`` if self is a shape optimization problem."""
        return self._is_shape_problem

    @is_shape_problem.setter
    def is_shape_problem(self, value: bool) -> None:
        """Setter method for is_shape_problem."""
        self._is_shape_problem = value
        self._is_control_problem = not value

    @property
    def is_control_problem(self) -> bool:
        """Returns a boolean flag, ``True`` if self is an optimal control problem."""
        return self._is_control_problem

    @is_control_problem.setter
    def is_control_problem(self, value: bool) -> None:
        """Setter method for is_control_problem."""
        self._is_shape_problem = not value
        self._is_control_problem = value

    @abc.abstractmethod
    def _erase_pde_memory(self) -> None:
        """Ensures that the PDEs are solved again, and no cache is used."""
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False

    def _parse_cost_functional_form(
        self,
        cost_functional_form: Union[
            List[types.CostFunctional], types.CostFunctional, List[ufl.Form], ufl.Form
        ],
    ) -> None:
        """Parses the cost functional form for use in cashocs."""
        self.input_cost_functional_list = _utils.enlist(cost_functional_form)
        self.cost_functional_list = []
        for functional in self.input_cost_functional_list:
            if isinstance(functional, ufl.Form):
                self.cost_functional_list.append(
                    cost_functional.IntegralFunctional(functional)
                )
            elif isinstance(
                functional,
                (
                    cost_functional.IntegralFunctional,
                    cost_functional.ScalarTrackingFunctional,
                    cost_functional.MinMaxFunctional,
                ),
            ):
                self.cost_functional_list.append(functional)

    def _parse_scalar_tracking_forms(
        self, scalar_tracking_forms: Optional[Union[List[Dict], Dict]]
    ) -> None:
        self.scalar_tracking_legacy_mode = False
        if scalar_tracking_forms is None:
            self.scalar_tracking_forms = scalar_tracking_forms
        else:
            _loggers.warning(
                "DEPRECATION WARNING: Using the keyword argument scalar_tracking_forms"
                " is replaced by using cashocs.ScalarTrackingFunctional and appending"
                "this to the list of cost functionals."
                "The former usage will be removed in the future."
            )
            self.scalar_tracking_forms = None

            for functional in _utils.enlist(scalar_tracking_forms):
                integrand = functional["integrand"]
                tracking_goal = functional["tracking_goal"]
                try:
                    weight = functional["weight"]
                except KeyError:
                    weight = 1.0
                scalar_tracking_functional = cost_functional.ScalarTrackingFunctional(
                    integrand, tracking_goal, weight
                )
                self.cost_functional_list.append(scalar_tracking_functional)
            self.use_scalar_tracking = False
            self.scalar_tracking_legacy_mode = True

    def _parse_optional_inputs(
        self,
        config: Optional[io.Config],
        initial_guess: Optional[Union[List[fenics.Function], fenics.Function]],
        ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ],
        adjoint_ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ],
        scalar_tracking_forms: Optional[Union[List[Dict], Dict]],
        min_max_terms: Optional[Union[List[Dict], Dict]],
        desired_weights: Optional[Union[List[float], float]],
    ) -> None:
        """Initializes the optional input parameters.

        Args:
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems.
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems.
            scalar_tracking_forms: A list of dictionaries that define scalar tracking
                type cost functionals, where an integral value should be brought to a
                desired value. Each dict needs to have the keys ``'integrand'`` and
                ``'tracking_goal'``.
            min_max_terms: Additional terms for the cost functional, not be used
                directly.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration.

        """
        if config is None:
            self.config = io.Config()
        else:
            self.config = copy.deepcopy(config)

        self.config.validate_config()

        if initial_guess is None:
            self.initial_guess = initial_guess
        else:
            self.initial_guess = _utils.enlist(initial_guess)

        if ksp_options is None:
            self.ksp_options: types.KspOptions = []
            option: List[List[Union[str, int, float]]] = copy.deepcopy(
                _utils.linalg.direct_ksp_options
            )

            for _ in range(self.state_dim):
                self.ksp_options.append(option)
        else:
            self.ksp_options = _utils.check_and_enlist_ksp_options(ksp_options)

        self.adjoint_ksp_options: types.KspOptions = (
            self.ksp_options[:]
            if adjoint_ksp_options is None
            else _utils.check_and_enlist_ksp_options(adjoint_ksp_options)
        )

        self._parse_scalar_tracking_forms(scalar_tracking_forms)

        if min_max_terms is None:
            self.min_max_terms = min_max_terms
        else:
            _loggers.warning(
                "DEPRECATION WARNING: Using the keyword argument min_max_terms"
                " is replaced by using cashocs.MinMaxFunctional and appending"
                "this to the list of cost functionals."
                "The former usage will be removed in the future."
            )
            self.min_max_terms = None
            self.use_min_max_terms = False
            for functional in _utils.enlist(min_max_terms):
                integrand = functional["integrand"]
                lower_bound = functional["lower_bound"]
                upper_bound = functional["upper_bound"]
                mu = functional["mu"]
                lambd = functional["lambda"]
                min_max_functional = cost_functional.MinMaxFunctional(
                    integrand, lower_bound, upper_bound, mu, lambd
                )
                self.cost_functional_list.append(min_max_functional)

        if desired_weights is None:
            self.desired_weights = desired_weights
        else:
            self.desired_weights = _utils.enlist(desired_weights)
            self.use_scaling = True

            if self.scalar_tracking_legacy_mode:
                raise _exceptions.InputError(
                    "OptimizationProblem",
                    "scalar_tracking_forms",
                    "Scaling of scalar_tracking_forms is now only possible when using "
                    "cashocs.ScalarTrackingFunctional",
                )

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
        adjoint_forms: Union[ufl.Form, List[ufl.Form]],
        adjoint_bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
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
        mod_bcs_list: List
        if adjoint_bcs_list == [] or adjoint_bcs_list is None:
            mod_bcs_list = []
            for i in range(self.state_dim):
                mod_bcs_list.append([])
        else:
            mod_bcs_list = _utils.check_and_enlist_bcs(adjoint_bcs_list)

        self.form_handler.bcs_list_ad = mod_bcs_list

        self.form_handler.adjoint_eq_forms = mod_forms
        # replace the adjoint function by a TrialFunction for internal use
        repl_forms = [
            ufl.replace(
                mod_forms[i],
                {self.adjoints[i]: self.form_handler.trial_functions_adjoint[i]},
            )
            for i in range(self.state_dim)
        ]
        self.form_handler.linear_adjoint_eq_forms = repl_forms

        (
            self.form_handler.adjoint_eq_lhs,
            self.form_handler.adjoint_eq_rhs,
        ) = _utils.split_linear_forms(self.form_handler.linear_adjoint_eq_forms)

        self.has_custom_adjoint = True

    def _check_for_custom_forms(self) -> None:
        """Checks, whether custom user forms are used and if they are compatible."""
        if self.has_custom_adjoint and not self.has_custom_derivative:
            _loggers.warning(
                "You only supplied the adjoint system. "
                "This might lead to unexpected results.\n"
                "Consider also supplying the (shape) derivative "
                "of the reduced cost functional,"
                "or check your approach with the cashocs.verification module."
            )

        elif not self.has_custom_adjoint and self.has_custom_derivative:
            _loggers.warning(
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

    def inject_pre_hook(self, function: Callable) -> None:
        """Changes the a-priori hook of the OptimizationProblem to function.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self.form_handler.pre_hook = function
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False

    def inject_post_hook(self, function: Callable) -> None:
        """Changes the a-posteriori hook of the OptimizationProblem to function.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self.form_handler.post_hook = function
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False

    def inject_pre_post_hook(
        self, pre_function: Callable, post_function: Callable
    ) -> None:
        """Changes the a-priori (pre) and a-posteriori (post) hook.

        Args:
            pre_function: A function without arguments, which is to be called before
                each solve of the state system
            post_function: A function without arguments, which is to be called after
                each computation of the (shape) gradient

        """
        self.inject_pre_hook(pre_function)
        self.inject_post_hook(post_function)

    @abc.abstractmethod
    def solve(
        self,
        algorithm: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
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
            self.config.set("OptimizationRoutine", "maximum_iterations", str(max_iter))

        self._check_for_custom_forms()
        self.output_manager = io.OutputManager(self)

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
        self.initial_function_values: List[float] = []
        for i, functional in enumerate(self.cost_functional_list):
            val = functional.evaluate()

            if abs(val) <= 1e-15:
                val = 1.0
                _loggers.info(
                    f"Term {i:d} of the cost functional vanishes "
                    f"for the initial iteration. Multiplying this term with the "
                    f"factor you supplied in desired weights."
                )

            self.initial_function_values.append(val)

    def _scale_cost_functional(self) -> None:
        """Scales the terms of the cost functional and scalar_tracking forms."""
        _loggers.info(
            "You are using the automatic scaling functionality of cashocs."
            "This may lead to unexpected results if you try to scale the cost "
            "functional yourself or if you supply custom forms."
        )

        if self.use_scaling and self.desired_weights is not None:
            if not self.has_cashocs_remesh_flag:
                self._compute_initial_function_values()

            else:
                with open(
                    f"{self.temp_dir}/temp_dict.json", "r", encoding="utf-8"
                ) as file:
                    temp_dict: Dict = json.load(file)
                self.initial_function_values = temp_dict["initial_function_values"]

            for i, functional in enumerate(self.cost_functional_list):
                scaling_factor = np.abs(
                    self.desired_weights[i] / self.initial_function_values[i]
                )
                functional.scale(scaling_factor)
                if isinstance(functional, cost_functional.IntegralFunctional):
                    self.input_cost_functional_list[i] = functional.form
