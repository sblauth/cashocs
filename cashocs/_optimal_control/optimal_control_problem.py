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

"""Class representing an optimal control problem.

"""

from __future__ import annotations

import configparser
from typing import Dict, List, Union, Optional

import fenics
import numpy as np
import ufl
from typing_extensions import Literal

from .methods import NCG, GradientDescent, LBFGS, Newton, PDAS
from .. import verification
from .._exceptions import InputError
from .._forms import ControlFormHandler
from .._interfaces import OptimizationProblem
from .._optimal_control import ReducedControlCostFunctional
from .._pde_problems import (
    AdjointProblem,
    GradientProblem,
    HessianProblem,
    StateProblem,
    UnconstrainedHessianProblem,
)
from ..utils import (
    _optimization_algorithm_configuration,
    enlist,
    _check_and_enlist_control_constraints,
)


class OptimalControlProblem(OptimizationProblem):
    """Implements an optimal control problem.

    This class is used to define an optimal control problem, and also to solve
    it subsequently. For a detailed documentation, see the examples in the :ref:`tutorial <tutorial_index>`.
    For easier input, when considering single (state or control) variables,
    these do not have to be wrapped into a list.
    Note, that in the case of multiple variables these have to be grouped into
    ordered lists, where state_forms, bcs_list, states, adjoints have to have
    the same order (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of ``y1``
    and so on.
    """

    def __new__(
        cls,
        state_forms: Union[ufl.Form, List[ufl.Form]],
        bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
        cost_functional_form: Union[ufl.Form, List[ufl.Form]],
        states: Union[fenics.Function, List[fenics.Function]],
        controls: Union[fenics.Function, List[fenics.Function]],
        adjoints: Union[fenics.Function, List[fenics.Function]],
        config: Optional[configparser.ConfigParser] = None,
        riesz_scalar_products: Optional[Union[ufl.Form, List[ufl.Form]]] = None,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[List[List[List[str]]]] = None,
        adjoint_ksp_options: Optional[List[List[List[str]]]] = None,
        scalar_tracking_forms: Optional[Union[Dict, List[Dict]]] = None,
        min_max_terms: Optional[List[Dict]] = None,
        desired_weights: Optional[List[float]] = None,
    ) -> OptimalControlProblem:

        try:
            if desired_weights is not None:
                _use_scaling = True
            else:
                _use_scaling = False
        except KeyError:
            _use_scaling = False

        if _use_scaling:
            unscaled_problem = super().__new__(cls)
            unscaled_problem.__init__(
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
            )
            unscaled_problem._scale_cost_functional()  # overwrites the cost functional list

        return super().__new__(cls)

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
        config: Optional[configparser.ConfigParser] = None,
        riesz_scalar_products: Optional[Union[ufl.Form, List[ufl.Form]]] = None,
        control_constraints: Optional[List[List[Union[float, fenics.Function]]]] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[List[List[List[str]]]] = None,
        adjoint_ksp_options: Optional[List[List[List[str]]]] = None,
        scalar_tracking_forms: Optional[Union[Dict, List[Dict]]] = None,
        min_max_terms: Optional[List[Dict]] = None,
        desired_weights: Optional[List[float]] = None,
    ) -> None:
        r"""This is used to generate all classes and functionalities. First ensures
        consistent input, afterwards, the solution algorithm is initialized.

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
        min_max_terms : dict or list[dict] or None, optional
            Additional terms for the cost functional, not to be used directly.
        desired_weights : list[float] or None, optional
            A list of values for scaling the cost functional terms. If this is supplied,
            the cost functional has to be given as list of summands. The individual terms
            are then scaled, so that term `i` has the magnitude of `desired_weights[i]`
            for the initial iteration. In case that `desired_weights` is `None`, no scaling
            is performed. Default is `None`.

        Examples
        --------
        Examples how to use this class can be found in the :ref:`tutorial <tutorial_index>`.
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

        self.controls = enlist(controls)
        self.control_dim = len(self.controls)

        ### riesz_scalar_products
        if riesz_scalar_products is None:
            dx = fenics.Measure("dx", self.controls[0].function_space().mesh())
            self.riesz_scalar_products = [
                fenics.inner(
                    fenics.TrialFunction(self.controls[i].function_space()),
                    fenics.TestFunction(self.controls[i].function_space()),
                )
                * dx
                for i in range(len(self.controls))
            ]
        else:
            self.riesz_scalar_products = enlist(riesz_scalar_products)
            self.uses_custom_scalar_product = True

        ### control_constraints
        if control_constraints is None:
            self.control_constraints = []
            for control in self.controls:
                u_a = fenics.Function(control.function_space())
                u_a.vector().vec().set(float("-inf"))
                u_b = fenics.Function(control.function_space())
                u_b.vector().vec().set(float("inf"))
                self.control_constraints.append([u_a, u_b])
        else:
            self.control_constraints = _check_and_enlist_control_constraints(
                control_constraints
            )

        # recast floats into functions for compatibility
        temp_constraints = self.control_constraints[:]
        self.control_constraints = []
        for idx, pair in enumerate(temp_constraints):
            if isinstance(pair[0], (float, int)):
                lower_bound = fenics.Function(self.controls[idx].function_space())
                lower_bound.vector().vec().set(pair[0])
            elif isinstance(pair[0], fenics.Function):
                lower_bound = pair[0]
            else:
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                    "control_constraints",
                    "Wrong type for the control constraints",
                )

            if isinstance(pair[1], (float, int)):
                upper_bound = fenics.Function(self.controls[idx].function_space())
                upper_bound.vector().vec().set(pair[1])
            elif isinstance(pair[1], fenics.Function):
                upper_bound = pair[1]
            else:
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                    "control_constraints",
                    "Wrong type for the control constraints",
                )

            self.control_constraints.append([lower_bound, upper_bound])

        ### Check whether the control constraints are feasible, and whether they are actually present
        self.require_control_constraints = [False for i in range(self.control_dim)]
        for idx, pair in enumerate(self.control_constraints):
            if not np.alltrue(pair[0].vector()[:] < pair[1].vector()[:]):
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                    "control_constraints",
                    "The lower bound must always be smaller than the upper bound for the control_constraints.",
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
                        if (
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
                            pass
                        else:
                            raise InputError(
                                "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                                "controls",
                                "Control constraints are only implemented for linear Lagrange, constant Discontinuous Lagrange, and Real elements.",
                            )

                else:
                    if (
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
                        pass
                    else:
                        raise InputError(
                            "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                            "controls",
                            "Control constraints are only implemented for linear Lagrange, constant Discontinuous Lagrange, and Real elements.",
                        )

        if not len(self.riesz_scalar_products) == self.control_dim:
            raise InputError(
                "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                "riesz_scalar_products",
                "Length of controls does not match",
            )
        if not len(self.control_constraints) == self.control_dim:
            raise InputError(
                "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem",
                "control_constraints",
                "Length of controls does not match",
            )
        ### end overloading

        self.form_handler = ControlFormHandler(self)

        self.state_spaces = self.form_handler.state_spaces
        self.control_spaces = self.form_handler.control_spaces
        self.adjoint_spaces = self.form_handler.adjoint_spaces

        self.projected_difference = [fenics.Function(V) for V in self.control_spaces]

        self.state_problem = StateProblem(self.form_handler, self.initial_guess)
        self.adjoint_problem = AdjointProblem(self.form_handler, self.state_problem)
        self.gradient_problem = GradientProblem(
            self.form_handler, self.state_problem, self.adjoint_problem
        )

        self.algorithm = _optimization_algorithm_configuration(self.config)

        self.reduced_cost_functional = ReducedControlCostFunctional(
            self.form_handler, self.state_problem
        )

        self.gradients = self.gradient_problem.gradients
        self.objective_value = 1.0

    def _erase_pde_memory(self) -> None:
        """Resets the memory of the PDE problems so that new solutions are computed.

        This sets the value of has_solution to False for all relevant PDE problems,
        where memory is stored.

        Returns
        -------
        None
        """

        super()._erase_pde_memory()
        self.gradient_problem.has_solution = False

    def solve(
        self,
        algorithm: Optional[
            Literal[
                "gradient_descent",
                "gd",
                "conjugate_gradient",
                "nonlinear_cg",
                "ncg",
                "lbfgs",
                "bfgs",
                "newton",
                "pdas",
                "primal_dual_active_set",
            ]
        ] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        r"""Solves the optimization problem by the method specified in the config file.

        Updates / overwrites states, controls, and adjoints according
        to the optimization method, i.e., the user-input :py:func:`fenics.Function` objects.

        Parameters
        ----------
        algorithm : str or None, optional
            Selects the optimization algorithm. Valid choices are
            ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
            ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
            for nonlinear conjugate gradient methods, ``'lbfgs'`` or ``'bfgs'`` for
            limited memory BFGS methods, ``'newton'`` for a truncated Newton method,
            and ``'pdas'`` or ``'primal_dual_active_set'`` for a
            primal dual active set method. This overwrites
            the value specified in the config file. If this is ``None``,
            then the value in the config file is used. Default is
            ``None``.
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

          .. math:: || \nabla J(u_k) || \leq \texttt{atol}.

          - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

          .. math:: || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol} || \nabla J(u_0) ||.
        """

        super().solve(algorithm=algorithm, rtol=rtol, atol=atol, max_iter=max_iter)

        if self.algorithm == "newton" or (
            self.algorithm == "pdas"
            and self.config.get("AlgoPDAS", "inner_pdas") == "newton"
        ):
            self.form_handler._ControlFormHandler__compute_newton_forms()

        if self.algorithm == "newton":
            self.hessian_problem = HessianProblem(
                self.form_handler, self.gradient_problem
            )
        if self.algorithm == "pdas":
            self.unconstrained_hessian = UnconstrainedHessianProblem(
                self.form_handler, self.gradient_problem
            )

        if self.algorithm == "gradient_descent":
            self.solver = GradientDescent(self)
        elif self.algorithm == "lbfgs":
            self.solver = LBFGS(self)
        elif self.algorithm == "conjugate_gradient":
            self.solver = NCG(self)
        elif self.algorithm == "newton":
            self.solver = Newton(self)
        elif self.algorithm == "pdas":
            self.solver = PDAS(self)
        elif self.algorithm == "none":
            raise InputError(
                "cashocs.OptimalControlProblem.solve",
                "algorithm",
                "You did not specify a solution algorithm in your config file. You have to specify one in the solve "
                "method. Needs to be one of"
                "'gradient_descent' ('gd'), 'lbfgs' ('bfgs'), 'conjugate_gradient' ('cg'), "
                "'newton', or 'primal_dual_active_set' ('pdas').",
            )

        self.solver.run()
        self.solver.post_processing()

    def compute_gradient(self) -> List[fenics.Function]:
        """Solves the Riesz problem to determine the gradient.

        This can be used for debugging, or code validation.
        The necessary solutions of the state and adjoint systems
        are carried out automatically.

        Returns
        -------
        list[fenics.Function]
            A list consisting of the (components) of the gradient.
        """

        self.gradient_problem.solve()

        return self.gradients

    def supply_derivatives(self, derivatives: Union[ufl.Form, List[ufl.Form]]) -> None:
        """Overwrites the derivatives of the reduced cost functional w.r.t. controls.

        This allows users to implement their own derivatives and use cashocs as a
        solver library only.

        Parameters
        ----------
        derivatives : ufl.Form or list[ufl.Form]
            The derivatives of the reduced (!) cost functional w.r.t. controls.

        Returns
        -------
        None
        """

        try:
            if isinstance(derivatives, list) and len(derivatives) > 0:
                for i in range(len(derivatives)):
                    if isinstance(derivatives[i], ufl.form.Form):
                        pass
                    else:
                        raise InputError(
                            "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                            "derivatives",
                            "derivatives have to be ufl forms",
                        )
                mod_derivatives = derivatives
            elif isinstance(derivatives, ufl.form.Form):
                mod_derivatives = [derivatives]
            else:
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                    "derivatives",
                    "derivatives have to be ufl forms",
                )
        except:
            raise InputError(
                "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                "derivatives",
                "derivatives have to be ufl forms",
            )

        for idx, form in enumerate(mod_derivatives):
            if len(form.arguments()) == 2:
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                    "derivatives",
                    "Do not use TrialFunction for the derivatives.",
                )
            elif len(form.arguments()) == 0:
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                    "derivatives",
                    "The specified derivatives must include a TestFunction object from the control space.",
                )

            if (
                not form.arguments()[0].ufl_function_space()
                == self.form_handler.control_spaces[idx]
            ):
                raise InputError(
                    "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                    "derivatives",
                    "The TestFunction has to be chosen from the same space as the corresponding adjoint.",
                )

        if not len(mod_derivatives) == self.form_handler.control_dim:
            raise InputError(
                "cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives",
                "derivatives",
                "Length of derivatives does not match number of controls.",
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

        This allows the user to specify both the derivatives of the reduced cost functional
        and the corresponding adjoint system, and allows them to use cashocs as a solver.

        See Also
        --------
        supply_derivatives
        supply_adjoint_forms

        Parameters
        ----------
        derivatives : ufl.Form or list[ufl.Form]
            The derivatives of the reduced (!) cost functional w.r.t. controls.
        adjoint_forms : ufl.Form or list[ufl.Form]
            The UFL forms of the adjoint system(s).
        adjoint_bcs_list : list[fenics.DirichletBC] or list[list[fenics.DirichletBC]] or fenics.DirichletBC or None
            The list of Dirichlet boundary conditions for the adjoint system(s).

        Returns
        -------
        None
        """

        self.supply_derivatives(derivatives)
        self.supply_adjoint_forms(adjoint_forms, adjoint_bcs_list)

    def gradient_test(
        self,
        u: Optional[List[fenics.Function]] = None,
        h: Optional[List[fenics.Function]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Taylor test to verify that the computed gradient is correct for optimal control problems.

        Parameters
        ----------
        u : list[fenics.Function] or None, optional
            The point, at which the gradient shall be verified. If this is ``None``,
            then the current controls of the optimization problem are used. Default is
            ``None``.
        h : list[fenics.Function] or None, optional
            The direction(s) for the directional (Gateaux) derivative. If this is ``None``,
            one random direction is chosen. Default is ``None``.
        rng : numpy.random.RandomState or None, optional
            A numpy random state for calculating a random direction

        Returns
        -------
        float
            The convergence order from the Taylor test. If this is (approximately) 2 or larger,
             everything works as expected.
        """

        return verification.control_gradient_test(self, u, h, rng)
