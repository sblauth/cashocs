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

"""Optimal control problem."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

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
from cashocs import verification
from cashocs._optimization import cost_functional
from cashocs._optimization import line_search as ls
from cashocs._optimization import optimal_control
from cashocs._optimization import optimization_algorithms
from cashocs._optimization import optimization_problem
from cashocs._optimization.optimal_control import box_constraints

if TYPE_CHECKING:
    from cashocs import _typing


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
        controls: list[fenics.Function] | fenics.Function,
        adjoints: list[fenics.Function] | fenics.Function,
        config: io.Config | None = None,
        riesz_scalar_products: list[ufl.Form] | ufl.Form | None = None,
        control_constraints: list[list[float | fenics.Function]] | None = None,
        initial_guess: list[fenics.Function] | None = None,
        ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        adjoint_ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        gradient_ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        desired_weights: list[float] | None = None,
        control_bcs_list: (
            list[list[fenics.DirichletBC]]
            | list[fenics.DirichletBC]
            | fenics.DirichletBC
            | None
        ) = None,
        preconditioner_forms: list[ufl.Form] | ufl.Form | None = None,
        pre_callback: Callable | None = None,
        post_callback: Callable | None = None,
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
            controls: The control variable(s), can either be a
                :py:class:`fenics.Function`, or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            config: The config file for the problem, generated via
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
            control_bcs_list: A list of boundary conditions for the control variables.
                This is passed analogously to ``bcs_list``. Default is ``None``.
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
            config=config,
            initial_guess=initial_guess,
            ksp_options=ksp_options,
            adjoint_ksp_options=adjoint_ksp_options,
            gradient_ksp_options=gradient_ksp_options,
            desired_weights=desired_weights,
            preconditioner_forms=preconditioner_forms,
            pre_callback=pre_callback,
            post_callback=post_callback,
            linear_solver=linear_solver,
            adjoint_linear_solver=adjoint_linear_solver,
            newton_linearizations=newton_linearizations,
            excluded_from_time_derivative=excluded_from_time_derivative,
        )

        self.db.function_db.controls = _utils.enlist(controls)

        self.db.parameter_db.control_dim = len(self.db.function_db.controls)
        self.db.function_db.control_spaces = [
            x.function_space() for x in self.db.function_db.controls
        ]
        self.db.function_db.gradient = _utils.create_function_list(
            self.db.function_db.control_spaces
        )
        self.db.parameter_db.problem_type = "control"

        self.mesh_parametrization = None

        self.riesz_scalar_products: list[ufl.Form] = self._parse_riesz_scalar_products(
            riesz_scalar_products
        )

        self.use_control_bcs = False
        self.control_bcs_list: list[list[fenics.DirichletBC]] | list[None]
        if control_bcs_list is not None:
            self.control_bcs_list_inhomogeneous = _utils.check_and_enlist_bcs(
                control_bcs_list
            )
            self.control_bcs_list = []
            for list_bcs in self.control_bcs_list_inhomogeneous:
                hom_bcs: list[fenics.DirichletBC] = [
                    fenics.DirichletBC(bc) for bc in list_bcs
                ]
                for bc in hom_bcs:
                    bc.homogenize()
                self.control_bcs_list.append(hom_bcs)  # type: ignore

            self.use_control_bcs = True
        else:
            self.control_bcs_list = [None] * len(self.db.function_db.controls)

        # control_constraints
        self.box_constraints = box_constraints.BoxConstraints(
            self.db.function_db.controls, control_constraints
        )
        self.db.parameter_db.display_box_constraints = (
            self.box_constraints.display_box_constraints
        )
        # end overloading

        self.is_control_problem = True
        self.form_handler: _forms.ControlFormHandler = _forms.ControlFormHandler(
            self, self.db
        )

        self.state_spaces = self.db.function_db.state_spaces
        self.adjoint_spaces = self.db.function_db.adjoint_spaces

        self.projected_difference = _utils.create_function_list(
            self.db.function_db.control_spaces
        )

        self.gradient_problem: _pde_problems.ControlGradientProblem = (
            _pde_problems.ControlGradientProblem(
                self.db, self.form_handler, self.state_problem, self.adjoint_problem
            )
        )

        self.reduced_cost_functional = cost_functional.ReducedCostFunctional(
            self.db, self.form_handler, self.state_problem
        )

        self.optimization_variable_abstractions = (
            optimal_control.ControlVariableAbstractions(
                self, self.box_constraints, self.db
            )
        )
        self._silent = False

        if bool(desired_weights is not None):
            self._scale_cost_functional()
            self.__init__(  # type: ignore
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
                gradient_ksp_options=gradient_ksp_options,
                desired_weights=None,
                control_bcs_list=control_bcs_list,
                preconditioner_forms=preconditioner_forms,
                pre_callback=pre_callback,
                post_callback=post_callback,
                linear_solver=linear_solver,
                adjoint_linear_solver=adjoint_linear_solver,
                newton_linearizations=newton_linearizations,
                excluded_from_time_derivative=excluded_from_time_derivative,
            )

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
            for i in range(len(self.db.function_db.controls)):
                for bc in self.control_bcs_list_inhomogeneous[i]:
                    bc.apply(self.db.function_db.controls[i].vector())

    def _setup_solver(self) -> optimization_algorithms.OptimizationAlgorithm:
        line_search_type = self.config.get("LineSearch", "method").casefold()
        if line_search_type == "armijo":
            line_search: ls.LineSearch = ls.ArmijoLineSearch(self.db, self)
        elif line_search_type == "polynomial":
            line_search = ls.PolynomialLineSearch(self.db, self)
        else:
            raise _exceptions.CashocsException("This code cannot be reached.")

        if self.algorithm.casefold() == "newton":
            self.form_handler.hessian_form_handler.compute_newton_forms()
            self.hessian_problem = _pde_problems.HessianProblem(
                self.db,
                self.form_handler,
                self.general_form_handler.adjoint_form_handler,
                self.gradient_problem,
                self.box_constraints,
            )

        if self.algorithm.casefold() == "gradient_descent":
            solver: optimization_algorithms.OptimizationAlgorithm = (
                optimization_algorithms.GradientDescentMethod(
                    self.db, self, line_search
                )
            )
        elif self.algorithm.casefold() == "lbfgs":
            solver = optimization_algorithms.LBFGSMethod(self.db, self, line_search)
        elif self.algorithm.casefold() == "conjugate_gradient":
            solver = optimization_algorithms.NonlinearCGMethod(
                self.db, self, line_search
            )
        elif self.algorithm.casefold() == "newton":
            solver = optimization_algorithms.NewtonMethod(self.db, self, line_search)
        elif self.algorithm.casefold() == "none":
            raise _exceptions.InputError(
                "cashocs.OptimalControlProblem.solve",
                "algorithm",
                "You did not specify a solution algorithm in your config file. "
                "You have to specify one in the solve method. Needs to be one of"
                "'gradient_descent' ('gd'), 'lbfgs' ('bfgs'), 'conjugate_gradient' "
                "('cg'), or 'newton'.",
            )
        else:
            raise _exceptions.CashocsException("This code cannot be reached.")

        return solver

    def solve(
        self,
        algorithm: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_iter: int | None = None,
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

            .. math::
                || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

            - a purely absolute one (if only ``atol`` is specified), i.e.,

            .. math::
                || \nabla J(u_K) || \leq \texttt{atol}.

            - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

            .. math::
                || \nabla J(u_k) || \leq \texttt{atol}
                + \texttt{rtol} || \nabla J(u_0) ||

        """
        if not self._silent:
            log.begin("Solving the optimal control problem.", level=log.INFO)
        super().solve(algorithm=algorithm, rtol=rtol, atol=atol, max_iter=max_iter)

        self._setup_control_bcs()

        self.solver = self._setup_solver()
        self.solver.run()
        self.solver.post_processing()
        if not self._silent:
            log.end()

    def compute_gradient(self) -> list[fenics.Function]:
        """Solves the Riesz problem to determine the gradient.

        This can be used for debugging, or code validation. The necessary solutions of
        the state and adjoint systems are carried out automatically.

        Returns:
            A list consisting of the (components) of the gradient.

        """
        self.gradient_problem.solve()

        return self.db.function_db.gradient

    def supply_derivatives(self, derivatives: ufl.Form | list[ufl.Form]) -> None:
        """Overwrites the derivatives of the reduced cost functional w.r.t. controls.

        This allows users to implement their own derivatives and use cashocs as a
        solver library only.

        Args:
            derivatives: The derivatives of the reduced (!) cost functional w.r.t.
            the control variables.

        """
        mod_derivatives: list[ufl.Form]
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
        derivatives: ufl.Form | list[ufl.Form],
        adjoint_forms: ufl.Form | list[ufl.Form],
        adjoint_bcs_list: (
            fenics.DirichletBC
            | list[fenics.DirichletBC]
            | list[list[fenics.DirichletBC]]
        ),
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
        u: list[fenics.Function] | None = None,
        h: list[fenics.Function] | None = None,
        rng: np.random.RandomState | None = None,
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
        self, riesz_scalar_products: list[ufl.Form] | ufl.Form
    ) -> list[ufl.Form]:
        """Checks, whether a given scalar product is symmetric.

        Args:
            riesz_scalar_products: The UFL forms of the scalar product.

        Returns:
            The (wrapped) list of scalar products

        """
        if riesz_scalar_products is None:
            dx = ufl.Measure(
                "dx", self.db.function_db.controls[0].function_space().mesh()
            )
            return [
                ufl.inner(
                    fenics.TrialFunction(self.db.function_db.control_spaces[i]),
                    fenics.TestFunction(self.db.function_db.control_spaces[i]),
                )
                * dx
                for i in range(len(self.db.function_db.controls))
            ]
        else:
            self.uses_custom_scalar_product = True
            return _utils.enlist(riesz_scalar_products)
