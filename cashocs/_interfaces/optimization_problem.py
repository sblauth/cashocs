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

"""Blueprints for the PDE constrained optimization problems.

This module is used to define the parent class for the optimization problems,
as many parameters and variables are common for optimal control and shape
optimization problems.
"""

from __future__ import annotations

import abc
import configparser
import copy
import json
from typing import Dict, List, Union, Optional, Callable

import fenics
import numpy as np
import ufl
from ufl import replace

from .._exceptions import InputError
from .._loggers import warning, info
from ..config import Config
from ..utils import (
    _parse_remesh,
    summation,
    _optimization_algorithm_configuration,
    enlist,
    _check_and_enlist_bcs,
    _check_and_enlist_ksp_options,
)


class OptimizationProblem(abc.ABC):
    """Blueprint for an abstract PDE constrained optimization problem.

    This class performs the initialization of the shared input so that the rest
    of CASHOCS can use it directly. Additionally, it includes methods that
    can be used to compute the state and adjoint variables by solving the
    corresponding equations. This could be subclassed to generate custom
    optimization problems.
    """

    def __init__(
        self,
        state_forms: Union[ufl.Form, List[ufl.Form]],
        bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
        cost_functional_form: Union[ufl.Form, List[ufl.Form]],
        states: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        config: Optional[configparser.ConfigParser] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[Union[List[List[str]]], List[List[List[str]]]] = None,
        adjoint_ksp_options: Optional[
            Union[List[List[str]]], List[List[List[str]]]
        ] = None,
        scalar_tracking_forms: Optional[Dict] = None,
        min_max_terms: Optional[Dict] = None,
        desired_weights: Optional[List[float]] = None,
    ) -> None:
        r"""Initializes the optimization problem.

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
        min_max_terms : dict or None, optional
            Additional terms for the cost functional, not be used directly.
        desired_weights : list[float] or None, optional
            A list of values for scaling the cost functional terms. If this is supplied,
            the cost functional has to be given as list of summands. The individual terms
            are then scaled, so that term `i` has the magnitude of `desired_weights[i]`
            for the initial iteration. In case that `desired_weights` is `None`, no scaling
            is performed. Default is `None`.

        Notes
        -----
        If one uses a single PDE constraint, the inputs can be the objects
        (UFL forms, functions, etc.) directly. In case multiple PDE constraints
        are present the inputs have to be put into (ordered) lists. The order of
        the objects depends on the order of the state variables, so that
        ``state_forms[i]`` is the weak form of the PDE for ``states[i]`` with boundary
        conditions ``bcs_list[i]`` and corresponding adjoint state ``adjoints[i]``.

        See Also
        --------
        cashocs.OptimalControlProblem : Represents an optimal control problem.
        cashocs.ShapeOptimizationProblem : Represents a shape optimization problem.
        """

        self.has_cashocs_remesh_flag, self.temp_dir = _parse_remesh()

        self.state_forms = enlist(state_forms)
        self.state_dim = len(self.state_forms)
        self.bcs_list = _check_and_enlist_bcs(bcs_list)

        self.cost_functional_list = enlist(cost_functional_form)
        self.cost_functional_form = summation(self.cost_functional_list)

        self.states = enlist(states)
        self.adjoints = enlist(adjoints)

        if config is None:
            self.config = Config()
            self.config.add_section("OptimizationRoutine")
            self.config.set("OptimizationRoutine", "algorithm", "none")
        else:
            self.config = copy.deepcopy(config)

        self.config.validate_config()

        if initial_guess is None:
            self.initial_guess = initial_guess
        else:
            self.initial_guess = enlist(initial_guess)

        if ksp_options is None:
            self.ksp_options = []
            option = [
                ["ksp_type", "preonly"],
                ["pc_type", "lu"],
                ["pc_factor_mat_solver_type", "mumps"],
                ["mat_mumps_icntl_24", 1],
            ]

            for i in range(self.state_dim):
                self.ksp_options.append(option)
        else:
            self.ksp_options = _check_and_enlist_ksp_options(ksp_options)

        if adjoint_ksp_options is None:
            self.adjoint_ksp_options = self.ksp_options[:]
        else:
            self.adjoint_ksp_options = _check_and_enlist_ksp_options(
                adjoint_ksp_options
            )

        self.use_scalar_tracking = False
        if scalar_tracking_forms is None:
            self.scalar_tracking_forms = scalar_tracking_forms
        else:
            self.scalar_tracking_forms = enlist(scalar_tracking_forms)
            self.use_scalar_tracking = True

        self.use_min_max_terms = False
        if min_max_terms is None:
            self.min_max_terms = min_max_terms
        else:
            self.use_min_max_terms = True
            self.min_max_terms = enlist(min_max_terms)

        self.use_scaling = False
        if desired_weights is None:
            self.desired_weights = desired_weights
        else:
            self.desired_weights = enlist(desired_weights)
            if isinstance(cost_functional_form, list):
                self.use_scaling = True
            else:
                raise InputError(
                    "OptimizationProblem",
                    "cost_functional_form",
                    "If you supply desired weights, you have to supply a matching list in cost_functional_form",
                )

            if scalar_tracking_forms is not None and not isinstance(
                scalar_tracking_forms, list
            ):
                raise InputError(
                    "OptimizationProblem",
                    "scalar_tracking_forms",
                    "If you supply desired weights, you have to supply a matching list in scalar_tracking_forms",
                )

        if not len(self.bcs_list) == self.state_dim:
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem",
                "bcs_list",
                "Length of states does not match.",
            )
        if not len(self.adjoints) == self.state_dim:
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem",
                "adjoints",
                "Length of states does not match.",
            )

        if self.initial_guess is not None:
            if not len(self.initial_guess) == self.state_dim:
                raise InputError(
                    "cashocs.optimization_problem.OptimizationProblem",
                    "initial_guess",
                    "Length of states does not match.",
                )

        if not len(self.ksp_options) == self.state_dim:
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem",
                "ksp_options",
                "Length of states does not match.",
            )
        if not len(self.adjoint_ksp_options) == self.state_dim:
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem",
                "ksp_options",
                "Length of states does not match.",
            )

        if self.desired_weights is not None:
            if not self.use_scalar_tracking:
                if not len(self.cost_functional_list) == len(self.desired_weights):
                    raise InputError(
                        "cashocs._interfaces.optimization_problem.OptimizationProblem",
                        "desired_weights",
                        "length of desired_weights does not fit",
                    )
            else:
                if not len(self.cost_functional_list) + len(
                    self.scalar_tracking_forms
                ) == len(self.desired_weights):
                    raise InputError(
                        "cashocs._interfaces.optimization_problem.OptimizationProblem",
                        "desired_weights",
                        "length of desired_weights does not fit",
                    )

        fenics.set_log_level(fenics.LogLevel.CRITICAL)

        self.state_problem = None
        self.adjoint_problem = None
        self.gradient_problem = None
        self.shape_gradient_problem = None

        self.algorithm = None

        self.form_handler = None
        self.has_custom_adjoint = False
        self.has_custom_derivative = False
        self.reduced_cost_functional = None
        self.uses_custom_scalar_product = False

    @abc.abstractmethod
    def _erase_pde_memory(self) -> None:
        """Ensures that the PDEs are solved again, and no cache is used

        Returns
        -------
        None
        """

        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False

    def compute_state_variables(self) -> None:
        """Solves the state system.

        This can be used for debugging purposes and to validate the solver.
        Updates and overwrites the user input for the state variables.

        Returns
        -------
        None
        """

        self.state_problem.solve()

    def compute_adjoint_variables(self) -> None:
        """Solves the adjoint system.

        This can be used for debugging purposes and solver validation.
        Updates / overwrites the user input for the adjoint variables.
        The solve of the corresponding state system needed to determine
        the adjoints is carried out automatically.

        Returns
        -------
        None
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

        This allows the user to specify their own weak forms of the problems and to use cashocs merely as
        a solver for solving the optimization problems.

        Parameters
        ----------
        adjoint_forms : ufl.Form or list[ufl.Form]
            The UFL forms of the adjoint system(s).
        adjoint_bcs_list : list[fenics.DirichletBC] or list[list[fenics.DirichletBC]] or fenics.DirichletBC or None
            The list of Dirichlet boundary conditions for the adjoint system(s).

        Returns
        -------
        None
        """

        mod_forms = enlist(adjoint_forms)

        if adjoint_bcs_list == [] or adjoint_bcs_list is None:
            mod_bcs_list = []
            for i in range(self.state_dim):
                mod_bcs_list.append([])
        else:
            mod_bcs_list = _check_and_enlist_bcs(adjoint_bcs_list)

        if not len(mod_forms) == self.form_handler.state_dim:
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem.supply_adjoint_forms",
                "adjoint_forms",
                "Length of adjoint_forms does not match",
            )
        if not len(mod_bcs_list) == self.form_handler.state_dim:
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem.supply_adjoint_forms",
                "adjoint_bcs_list",
                "Length of adjoint_bcs_list does not match",
            )

        for idx, form in enumerate(mod_forms):
            if len(form.arguments()) == 2:
                raise InputError(
                    "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms",
                    "adjoint_forms",
                    "Do not use TrialFunction for the adjoints, but the actual Function you passed to th OptimalControlProblem.",
                )
            elif len(form.arguments()) == 0:
                raise InputError(
                    "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms",
                    "adjoint_forms",
                    "The specified adjoint_forms must include a TestFunction object.",
                )

            if (
                not form.arguments()[0].ufl_function_space()
                == self.form_handler.adjoint_spaces[idx]
            ):
                raise InputError(
                    "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms",
                    "adjoint_forms",
                    "The TestFunction has to be chosen from the same space as the corresponding adjoint.",
                )

        self.form_handler.adjoint_picard_forms = mod_forms
        self.form_handler.bcs_list_ad = mod_bcs_list

        # replace the adjoint function by a TrialFunction for internal use
        repl_forms = [
            replace(
                mod_forms[i],
                {self.adjoints[i]: self.form_handler.trial_functions_adjoint[i]},
            )
            for i in range(self.state_dim)
        ]
        self.form_handler.adjoint_eq_forms = repl_forms

        self.form_handler.adjoint_eq_lhs = []
        self.form_handler.adjoint_eq_rhs = []

        for i in range(self.state_dim):
            a, L = fenics.system(self.form_handler.adjoint_eq_forms[i])
            self.form_handler.adjoint_eq_lhs.append(a)
            if L.empty():
                zero_form = (
                    fenics.inner(
                        fenics.Constant(
                            np.zeros(
                                self.form_handler.test_functions_adjoint[i].ufl_shape
                            )
                        ),
                        self.form_handler.test_functions_adjoint[i],
                    )
                    * self.form_handler.dx
                )
                self.form_handler.adjoint_eq_rhs.append(zero_form)
            else:
                self.form_handler.adjoint_eq_rhs.append(L)

        self.has_custom_adjoint = True

    def _check_for_custom_forms(self) -> None:
        """Checks whether custom user forms are used and if they are compatible with the settings.

        Returns
        -------
        None
        """

        if self.has_custom_adjoint and not self.has_custom_derivative:
            warning(
                "You only supplied the adjoint system. This might lead to unexpected results.\n"
                "Consider also supplying the (shape) derivative of the reduced cost functional,"
                "or check your approach with the cashocs.verification module."
            )

        elif not self.has_custom_adjoint and self.has_custom_derivative:
            warning(
                "You only supplied the derivative of the reduced cost functional. This might lead to unexpected results.\n"
                "Consider also supplying the adjoint system, "
                "or check your approach with the cashocs.verification module."
            )

        if self.algorithm == "newton" and (
            self.has_custom_adjoint or self.has_custom_derivative
        ):
            raise InputError(
                "cashocs.optimization_problem.OptimizationProblem",
                "solve",
                "The usage of custom forms is not compatible with the Newton solver."
                "Please do not supply custom forms if you want to use the Newton solver.",
            )

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

        self.form_handler._pre_hook = function
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        try:
            self.gradient_problem.has_solution = False
        except AttributeError:
            self.shape_gradient_problem.has_solution = False

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

        self.form_handler._post_hook = function
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        try:
            self.gradient_problem.has_solution = False
        except AttributeError:
            self.shape_gradient_problem.has_solution = False

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

    @abc.abstractmethod
    def solve(
        self,
        algorithm: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        r"""Solves the optimization problem by the method specified in the config file.

        Parameters
        ----------
        algorithm : str or None, optional
            Selects the optimization algorithm. Valid choices are
            ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
            ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
            for nonlinear conjugate gradient methods, and ``'lbfgs'`` or ``'bfgs'`` for
            limited memory BFGS methods. This overwrites the value specified
            in the config file. If this is ``None``, then the value in the
            config file is used. Default is ``None``. In addition, for optimal control problems,
            one can use ``'newton'`` for a truncated Newton method,
            and ``'pdas'`` or ``'primal_dual_active_set'`` for a
            primal dual active set method.
        rtol : float or None, optional
            The relative tolerance used for the termination criterion.
            Overwrites the value specified in the config file. If this
            is ``None``, the value from the config file is taken. Default
            is ``None``.
        atol : float or None, optional
            The absolute tolerance used for the termination criterion.
            Overwrites the value specified in the config file. If this
            is ``None``, the value from the config file is taken. Default
            is ``None``.
        max_iter : int or None, optional
            The maximum number of iterations the optimization algorithm
            can carry out before it is terminated. Overwrites the value
            specified in the config file. If this is ``None``, the value from
            the config file is taken. Default is ``None``.

        Returns
        -------
        None

        Notes
        -----
        If either ``rtol`` or ``atol`` are specified as arguments to the solve
        call, the termination criterion changes to:

          - a purely relative one (if only ``rtol`` is specified), i.e.,

          .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

          - a purely absolute one (if only ``atol`` is specified), i.e.,

          .. math:: || \nabla J(u_K) || \leq \texttt{atol}.

          - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

          .. math:: || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol} || \nabla J(u_0) ||
        """

        self.algorithm = _optimization_algorithm_configuration(self.config, algorithm)

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

    def __shift_cost_functional(self, shift: float = 0.0) -> None:
        """Shifts the cost functional by a constant.

        Parameters
        ----------
        shift : float, optional
            The constant, by which the cost functional is shifted.

        Returns
        -------
        None
        """

        self.form_handler.cost_functional_shift = shift

    @abc.abstractmethod
    def gradient_test(self):
        """Test the computed gradient with finite differences.

        Returns
        -------
        float
            The result of the gradient test. If this is (approximately) 2 or larger,
            everything works as expected.
        """
        pass

    def _scale_cost_functional(self):
        """Scales the terms of the cost functional and scalar_tracking forms.

        Returns
        -------
        None
        """

        info(
            "You are using the automatic scaling functionality of cashocs."
            "This may lead to unexpected results if you try to scale the cost functional yourself or if you supply custom forms."
        )

        if not self.has_cashocs_remesh_flag:
            self.state_problem.solve()
            self.initial_function_values = []
            for i in range(len(self.cost_functional_list)):
                val = fenics.assemble(self.cost_functional_list[i])

                if abs(val) <= 1e-15:
                    val = 1.0
                    info(
                        f"Term {i:d} of the cost functional vanishes for the initial iteration. Multiplying this term with the factor you supplied in desired weights."
                    )

                self.initial_function_values.append(val)

            if self.use_scalar_tracking:
                self.initial_scalar_tracking_values = []
                for i in range(len(self.scalar_tracking_forms)):
                    val = 0.5 * pow(
                        fenics.assemble(
                            self.form_handler.scalar_cost_functional_integrands[i]
                        )
                        - self.form_handler.scalar_tracking_goals[i],
                        2,
                    )

                    if abs(val) <= 1e-15:
                        val = 1.0
                        info(
                            f"Term {i:d} of the scalar tracking cost functional vanishes for the initial iteration. Multiplying this term with the factor you supplied in desired weights."
                        )

                    self.initial_scalar_tracking_values.append(val)

        else:
            with open(f"{self.temp_dir}/temp_dict.json", "r") as file:
                temp_dict = json.load(file)
            self.initial_function_values = temp_dict["initial_function_values"]
            if self.use_scalar_tracking:
                self.initial_scalar_tracking_values = temp_dict[
                    "initial_scalar_tracking_values"
                ]

        for i in range(len(self.cost_functional_list)):
            self.cost_functional_list[i] = (
                fenics.Constant(
                    abs(self.desired_weights[i] / self.initial_function_values[i])
                )
                * self.cost_functional_list[i]
            )

        if self.use_scalar_tracking:
            for i in range(len(self.scalar_tracking_forms)):
                self.scalar_tracking_forms[-1 - i]["weight"] = abs(
                    self.desired_weights[-1 - i]
                    / self.initial_scalar_tracking_values[-1 - i]
                )
