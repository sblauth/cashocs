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

"""Class representing an optimal control problem."""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
import ufl

from cashocs import _exceptions
from cashocs import _forms
from cashocs import _pde_problems
from cashocs import _utils
from cashocs._optimization import cost_functional
from cashocs._optimization import line_search
from cashocs._optimization import optimal_control
from cashocs._optimization import optimization_algorithms
from cashocs._optimization import optimization_problem
from cashocs._optimization import verification

if TYPE_CHECKING:
    from cashocs import io
    from cashocs import types


class OptimalControlProblem(optimization_problem.OptimizationProblem):
    """Implements an optimal control problem.

    This class is used to define an optimal control problem, and also to solve
    it subsequently. For a detailed documentation, see the examples in the
    :ref:`tutorial <tutorial_index>`. For easier input, when considering single
    (state or control) variables, these do not have to be wrapped into a list.
    Note, that in the case of multiple variables these have to be grouped into
    ordered lists, where state_forms, bcs_list, states, adjoints have to have
    the same order (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of
    ``y1`` and so on).
    """

    controls: List[fenics.Function]
    control_spaces: List[fenics.FunctionSpace]

    def __new__(
        cls,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            List[List[fenics.DirichletBC]], List[fenics.DirichletBC], fenics.DirichletBC
        ],
        cost_functional_form: Union[
            List[types.CostFunctional], types.CostFunctional, List[ufl.Form], ufl.Form
        ],
        states: Union[List[fenics.Function], fenics.Function],
        controls: Union[List[fenics.Function], fenics.Function],
        adjoints: Union[List[fenics.Function], fenics.Function],
        config: Optional[io.Config] = None,
        riesz_scalar_products: Optional[Union[List[ufl.Form], ufl.Form]] = None,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
        min_max_terms: Optional[Union[List[Dict], Dict]] = None,
        desired_weights: Optional[List[float]] = None,
        control_bcs_list: Optional[
            Union[
                List[List[fenics.DirichletBC]],
                List[fenics.DirichletBC],
                fenics.DirichletBC,
            ]
        ] = None,
    ) -> OptimalControlProblem:
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
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
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
            scalar_tracking_forms: A list of dictionaries that define scalar tracking
                type cost functionals, where an integral value should be brought to a
                desired value. Each dict needs to have the keys ``'integrand'`` and
                ``'tracking_goal'``. Default is ``None``, i.e., no scalar tracking terms
                are considered.
            min_max_terms: Additional terms for the cost functional, not to be used
                directly.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.
            control_bcs_list: A list of boundary conditions for the control variables.
                This is passed analogously to ``bcs_list``. Default is ``None``.

        Examples:
            Examples how to use this class can be found in the :ref:`tutorial
            <tutorial_index>`.

        """
        use_scaling = bool(desired_weights is not None)

        if use_scaling:
            unscaled_problem = super().__new__(cls)
            unscaled_problem.__init__(  # type: ignore
                state_forms,
                bcs_list,
                cost_functional_form,
                states,
                controls,
                adjoints,
                config=config,
                riesz_scalar_products=riesz_scalar_products,
                control_constraints=control_constraints,
                initial_guess=initial_guess,
                ksp_options=ksp_options,
                adjoint_ksp_options=adjoint_ksp_options,
                scalar_tracking_forms=scalar_tracking_forms,
                min_max_terms=min_max_terms,
                desired_weights=desired_weights,
                control_bcs_list=control_bcs_list,
            )
            unscaled_problem._scale_cost_functional()  # overwrites cost functional list

        return super().__new__(cls)

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
        controls: Union[List[fenics.Function], fenics.Function],
        adjoints: Union[List[fenics.Function], fenics.Function],
        config: Optional[io.Config] = None,
        riesz_scalar_products: Optional[Union[List[ufl.Form], ufl.Form]] = None,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
        min_max_terms: Optional[Union[List[Dict], Dict]] = None,
        desired_weights: Optional[List[float]] = None,
        control_bcs_list: Optional[
            Union[
                List[List[fenics.DirichletBC]],
                List[fenics.DirichletBC],
                fenics.DirichletBC,
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
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
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
            scalar_tracking_forms: A list of dictionaries that define scalar tracking
                type cost functionals, where an integral value should be brought to a
                desired value. Each dict needs to have the keys ``'integrand'`` and
                ``'tracking_goal'``. Default is ``None``, i.e., no scalar tracking terms
                are considered.
            min_max_terms: Additional terms for the cost functional, not to be used
                directly.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.
            control_bcs_list: A list of boundary conditions for the control variables.
                This is passed analogously to ``bcs_list``. Default is ``None``.

        Examples:
            Examples how to use this class can be found in the :ref:`tutorial
            <tutorial_index>`.

        """
        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            config,
            initial_guess,
            ksp_options,
            adjoint_ksp_options,
            scalar_tracking_forms,
            min_max_terms,
            desired_weights,
        )

        self.controls = _utils.enlist(controls)
        self.control_dim = len(self.controls)

        # riesz_scalar_products
        self.riesz_scalar_products = self._parse_riesz_scalar_products(
            riesz_scalar_products
        )

        self.use_control_bcs = False
        self.control_bcs_list: Union[List[List[fenics.DirichletBC]], List[None]]
        if control_bcs_list is not None:
            self.control_bcs_list_inhomogeneous = _utils.check_and_enlist_bcs(
                control_bcs_list
            )
            self.control_bcs_list = []  # type: ignore
            for list_bcs in self.control_bcs_list_inhomogeneous:
                hom_bcs: List[fenics.DirichletBC] = [
                    fenics.DirichletBC(bc) for bc in list_bcs
                ]
                for bc in hom_bcs:
                    bc.homogenize()
                self.control_bcs_list.append(hom_bcs)  # type: ignore

            self.use_control_bcs = True
        else:
            self.control_bcs_list = [None] * self.control_dim

        # control_constraints
        self.control_constraints = self._parse_control_constraints(control_constraints)
        self._validate_control_constraints()

        # end overloading

        self.is_control_problem = True
        self.form_handler: _forms.ControlFormHandler = _forms.ControlFormHandler(self)

        self.state_spaces = self.form_handler.state_spaces
        self.control_spaces = self.form_handler.control_spaces
        self.adjoint_spaces = self.form_handler.adjoint_spaces

        self.projected_difference = _utils.create_function_list(self.control_spaces)

        self.state_problem = _pde_problems.StateProblem(
            self.form_handler, self.initial_guess
        )
        self.adjoint_problem = _pde_problems.AdjointProblem(
            self.form_handler, self.state_problem
        )
        self.gradient_problem: _pde_problems.ControlGradientProblem = (
            _pde_problems.ControlGradientProblem(
                self.form_handler, self.state_problem, self.adjoint_problem
            )
        )

        self.algorithm = _utils.optimization_algorithm_configuration(self.config)

        self.reduced_cost_functional = cost_functional.ReducedCostFunctional(
            self.form_handler, self.state_problem
        )

        self.gradient = self.gradient_problem.gradient
        self.objective_value = 1.0

    def _erase_pde_memory(self) -> None:
        """Resets the memory of the PDE problems so that new solutions are computed.

        This sets the value of has_solution to False for all relevant PDE problems,
        where memory is stored.
        """
        super()._erase_pde_memory()
        self.gradient_problem.has_solution = False

    def _setup_control_bcs(self) -> None:
        """Sets up the boundary conditions for the control variables."""
        if self.use_control_bcs:
            for i in range(self.control_dim):
                for bc in self.control_bcs_list_inhomogeneous[i]:
                    bc.apply(self.controls[i].vector())

    def solve(
        self,
        algorithm: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        r"""Solves the optimization problem by the method specified in the config file.

        Updates / overwrites states, controls, and adjoints according
        to the optimization method, i.e., the user-input :py:func:`fenics.Function`
        objects.

        Args:
            algorithm: Selects the optimization algorithm. Valid choices are
                ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
                ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
                for nonlinear conjugate gradient methods, ``'lbfgs'`` or ``'bfgs'`` for
                limited memory BFGS methods, and ``'newton'`` for a truncated Newton
                method. This overwrites the value specified in the config file. If this
                is ``None``, then the value in the config file is used. Default is
                ``None``.
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
        super().solve(algorithm=algorithm, rtol=rtol, atol=atol, max_iter=max_iter)

        self._setup_control_bcs()

        self.optimization_variable_abstractions = (
            optimal_control.ControlVariableAbstractions(self)
        )
        self.line_search = line_search.ArmijoLineSearch(self)

        if self.algorithm.casefold() == "newton":
            self.form_handler.compute_newton_forms()

        if self.algorithm.casefold() == "newton":
            self.hessian_problem = _pde_problems.HessianProblem(
                self.form_handler, self.gradient_problem
            )

        if self.algorithm.casefold() == "gradient_descent":
            self.solver = optimization_algorithms.GradientDescentMethod(
                self, self.line_search
            )
        elif self.algorithm.casefold() == "lbfgs":
            self.solver = optimization_algorithms.LBFGSMethod(self, self.line_search)
        elif self.algorithm.casefold() == "conjugate_gradient":
            self.solver = optimization_algorithms.NonlinearCGMethod(
                self, self.line_search
            )
        elif self.algorithm.casefold() == "newton":
            self.solver = optimization_algorithms.NewtonMethod(self, self.line_search)
        elif self.algorithm.casefold() == "none":
            raise _exceptions.InputError(
                "cashocs.OptimalControlProblem.solve",
                "algorithm",
                "You did not specify a solution algorithm in your config file. "
                "You have to specify one in the solve method. Needs to be one of"
                "'gradient_descent' ('gd'), 'lbfgs' ('bfgs'), 'conjugate_gradient' "
                "('cg'), or 'newton'.",
            )

        self.solver.run()
        self.solver.post_processing()

    def compute_gradient(self) -> List[fenics.Function]:
        """Solves the Riesz problem to determine the gradient.

        This can be used for debugging, or code validation. The necessary solutions of
        the state and adjoint systems are carried out automatically.

        Returns:
            A list consisting of the (components) of the gradient.

        """
        self.gradient_problem.solve()

        return self.gradient

    def supply_derivatives(self, derivatives: Union[ufl.Form, List[ufl.Form]]) -> None:
        """Overwrites the derivatives of the reduced cost functional w.r.t. controls.

        This allows users to implement their own derivatives and use cashocs as a
        solver library only.

        Args:
            derivatives: The derivatives of the reduced (!) cost functional w.r.t.
            the control variables.

        """
        mod_derivatives: List[ufl.Form]
        if isinstance(derivatives, ufl.form.Form):
            mod_derivatives = [derivatives]
        else:
            mod_derivatives = derivatives

        self.form_handler.setup_assemblers(
            self.form_handler.riesz_scalar_products,
            mod_derivatives,
            self.form_handler.control_bcs_list,
        )

        self.form_handler.gradient_forms_rhs = mod_derivatives
        self.has_custom_derivative = True

    def supply_custom_forms(
        self,
        derivatives: Union[ufl.Form, List[ufl.Form]],
        adjoint_forms: Union[ufl.Form, List[ufl.Form]],
        adjoint_bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
    ) -> None:
        """Overrides both adjoint system and derivatives with user input.

        This allows the user to specify both the derivatives of the reduced cost
        functional and the corresponding adjoint system, and allows them to use cashocs
        as a solver.

        Args:
            derivatives: The derivatives of the reduced (!) cost functional w.r.t. the
                control variables.
            adjoint_forms: The UFL forms of the adjoint system(s).
            adjoint_bcs_list: The list of Dirichlet boundary conditions for the adjoint
                system(s).

        """
        self.supply_derivatives(derivatives)
        self.supply_adjoint_forms(adjoint_forms, adjoint_bcs_list)

    def gradient_test(
        self,
        u: Optional[List[fenics.Function]] = None,
        h: Optional[List[fenics.Function]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Performs a Taylor test to verify correctness of the computed gradient.

        Args:
            u: The point, at which the gradient shall be verified. If this is ``None``,
                then the current controls of the optimization problem are used. Default
                is ``None``.
            h: The direction(s) for the directional (Gâteaux) derivative. If this is
                ``None``, one random direction is chosen. Default is ``None``.
            rng: A numpy random state for calculating a random direction.

        Returns:
            The convergence order from the Taylor test. If this is (approximately) 2
            or larger, everything works as expected.

        """
        return verification.control_gradient_test(self, u, h, rng)

    def _parse_riesz_scalar_products(
        self, riesz_scalar_products: Union[List[ufl.Form], ufl.Form]
    ) -> List[ufl.Form]:
        """Checks, whether a given scalar product is symmetric.

        Args:
            riesz_scalar_products: The UFL forms of the scalar product.

        Returns:
            The (wrapped) list of scalar products

        """
        if riesz_scalar_products is None:
            dx = fenics.Measure("dx", self.controls[0].function_space().mesh())
            return [
                fenics.inner(
                    fenics.TrialFunction(self.controls[i].function_space()),
                    fenics.TestFunction(self.controls[i].function_space()),
                )
                * dx
                for i in range(len(self.controls))
            ]
        else:
            self.uses_custom_scalar_product = True
            return _utils.enlist(riesz_scalar_products)

    def _parse_control_constraints(
        self,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]],
    ) -> List[List[fenics.Function]]:
        """Checks, whether the given control constraints are feasible.

        Args:
            control_constraints: The control constraints.

        Returns:
            The (wrapped) list of control constraints.

        """
        temp_control_constraints: List[List[Union[fenics.Function, float]]]
        if control_constraints is None:
            temp_control_constraints = []
            for control in self.controls:
                u_a = fenics.Function(control.function_space())
                u_a.vector().vec().set(float("-inf"))
                u_a.vector().apply("")
                u_b = fenics.Function(control.function_space())
                u_b.vector().vec().set(float("inf"))
                u_b.vector().apply("")
                temp_control_constraints.append([u_a, u_b])
        else:
            temp_control_constraints = _utils.check_and_enlist_control_constraints(
                control_constraints
            )

        # recast floats into functions for compatibility
        formatted_control_constraints: List[List[fenics.Function]] = []
        for idx, pair in enumerate(temp_control_constraints):
            if isinstance(pair[0], fenics.Function):
                lower_bound = pair[0]
            else:
                lower_bound = fenics.Function(self.controls[idx].function_space())
                lower_bound.vector().vec().set(pair[0])
                lower_bound.vector().apply("")

            if isinstance(pair[1], fenics.Function):
                upper_bound = pair[1]
            else:
                upper_bound = fenics.Function(self.controls[idx].function_space())
                upper_bound.vector().vec().set(pair[1])
                upper_bound.vector().apply("")

            formatted_control_constraints.append([lower_bound, upper_bound])

        return formatted_control_constraints

    def _validate_control_constraints(self) -> None:
        """Checks, whether given control constraints are valid."""
        self.require_control_constraints = [False] * self.control_dim
        for idx, pair in enumerate(self.control_constraints):
            if not np.all(pair[0].vector()[:] < pair[1].vector()[:]):
                raise _exceptions.InputError(
                    (
                        "cashocs._optimization.optimal_control."
                        "optimal_control_problem.OptimalControlProblem"
                    ),
                    "control_constraints",
                    (
                        "The lower bound must always be smaller than the upper bound "
                        "for the control_constraints."
                    ),
                )

            if pair[0].vector().vec().max()[1] == float("-inf") and pair[
                1
            ].vector().vec().min()[1] == float("inf"):
                # no control constraint for this component
                pass
            else:
                self.require_control_constraints[idx] = True

                control_element = self.controls[idx].ufl_element()
                if control_element.family() == "Mixed":
                    for j in range(control_element.value_size()):
                        sub_elem = control_element.extract_component(j)[1]
                        if not (
                            sub_elem.family() == "Real"
                            or (
                                sub_elem.family() == "Lagrange"
                                and sub_elem.degree() == 1
                            )
                            or (
                                sub_elem.family() == "Discontinuous Lagrange"
                                and sub_elem.degree() == 0
                            )
                        ):
                            raise _exceptions.InputError(
                                (
                                    "cashocs._optimization.optimal_control."
                                    "optimal_control_problem.OptimalControlProblem"
                                ),
                                "controls",
                                (
                                    "Control constraints are only implemented for "
                                    "linear Lagrange, constant Discontinuous Lagrange, "
                                    "and Real elements."
                                ),
                            )

                else:
                    if not (
                        control_element.family() == "Real"
                        or (
                            control_element.family() == "Lagrange"
                            and control_element.degree() == 1
                        )
                        or (
                            control_element.family() == "Discontinuous Lagrange"
                            and control_element.degree() == 0
                        )
                    ):
                        raise _exceptions.InputError(
                            (
                                "cashocs._optimization.optimal_control."
                                "optimal_control_problem.OptimalControlProblem"
                            ),
                            "controls",
                            (
                                "Control constraints are only implemented for "
                                "linear Lagrange, constant Discontinuous Lagrange, "
                                "and Real elements."
                            ),
                        )
