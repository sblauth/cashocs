from __future__ import annotations

import abc
import copy
from typing import Callable, List, Optional, TYPE_CHECKING, Union

import fenics
from matplotlib import colors
import matplotlib.pyplot as pp
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _optimization
from cashocs import _utils
from cashocs import io
from cashocs._optimization import cost_functional
from cashocs._optimization import line_search as ls
from cashocs._optimization import topology_optimization
from cashocs._optimization.optimal_control import optimal_control_problem
from cashocs._optimization.topology_optimization import bisection
from cashocs._optimization.topology_optimization import descent_topology_algorithm
from cashocs._optimization.topology_optimization import topology_optimization_algorithm
from cashocs._optimization.topology_optimization import topology_optimization_problem
from cashocs._optimization.topology_optimization import topology_variable_abstractions

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs import _pde_problems
    from cashocs import _typing


class DeflatedProblem(abc.ABC):

    def __init__(  # pylint: disable=unused-argument
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
        initial_guess: list[fenics.Function] | None = None,
        ksp_options: Optional[Union[_typing.KspOption, List[_typing.KspOption]]] = None,
        adjoint_ksp_options: Optional[
            Union[_typing.KspOption, List[_typing.KspOption]]
        ] = None,
        preconditioner_forms: Optional[Union[List[ufl.Form], ufl.Form]] = None,
    ) -> None:
        r"""Initializes the topology optimization problem.

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
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.

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
        self.preconditioner_forms = preconditioner_forms

        self.initial_norm = 0.0

        self._pre_callback: Optional[
            Union[Callable[[], None], Callable[[_typing.OptimizationProblem], None]]
        ] = None
        self._post_callback: Optional[
            Union[Callable[[], None], Callable[[_typing.OptimizationProblem], None]]
        ] = None

        self.cost_functional_form_initial: List[_typing.CostFunctional] = _utils.enlist(
            cost_functional_form
        )

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

    def inject_pre_callback(
        self,
        function: Optional[
            Union[Callable[[], None], Callable[[_typing.OptimizationProblem], None]]
        ],
    ) -> None:
        """Changes the a-priori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self._pre_callback = function

    def inject_post_callback(
        self,
        function: Optional[
            Union[Callable[[], None], Callable[[_typing.OptimizationProblem], None]]
        ],
    ) -> None:
        """Changes the a-posteriori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self._post_callback = function

    def inject_pre_post_callback(
        self,
        pre_function: Optional[
            Union[Callable[[], None], Callable[[_typing.OptimizationProblem], None]]
        ],
        post_function: Optional[
            Union[Callable[[], None], Callable[[_typing.OptimizationProblem], None]]
        ],
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


class DeflatedTopologyOptimizationProblem(DeflatedProblem):

    def __init__(  # pylint: disable=unused-argument
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
        levelset_function: fenics.Function,
        topological_derivative_neg: fenics.Function | ufl.Form,
        topological_derivative_pos: fenics.Function | ufl.Form,
        update_levelset: Callable,
        volume_restriction: Union[float, tuple[float, float]] | None = None,
        config: io.Config | None = None,
        riesz_scalar_products: list[ufl.Form] | ufl.Form | None = None,
        initial_guess: list[fenics.Function] | None = None,
        ksp_options: Optional[Union[_typing.KspOption, List[_typing.KspOption]]] = None,
        adjoint_ksp_options: Optional[
            Union[_typing.KspOption, List[_typing.KspOption]]
        ] = None,
        gradient_ksp_options: Optional[
            Union[_typing.KspOption, List[_typing.KspOption]]
        ] = None,
        preconditioner_forms: Optional[Union[List[ufl.Form], ufl.Form]] = None,
    ) -> None:
        r"""Initializes the deflated topology optimization problem.

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
            levelset_function: A :py:class:`fenics.Function` which represents the
                levelset function.
            topological_derivative_neg: The topological derivative inside the domain,
                where the levelset function is negative.
            topological_derivative_pos: The topological derivative inside the domain,
                where the levelset function is positive.
            update_levelset: A python function (without arguments) which is called to
                update the coefficients etc. when the levelset function is changed.
            volume_restriction: A volume restriction for the optimization problem.
                A single floats describes an equality constraint and a tuple of floats
                an inequality constraint.
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            riesz_scalar_products: The scalar products of the control space. Can either
                be ``None`` or a single UFL form. If it is ``None``, the
                :math:`L^2(\Omega)` product is used (default is ``None``).
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
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.

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
            preconditioner_forms=preconditioner_forms,
        )

        self.levelset_function: fenics.Function = levelset_function
        self.levelset_function_init = fenics.Function(
            self.levelset_function.function_space()
        )
        self.levelset_function_init.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_init.vector().apply("")

        self.topological_derivative_pos: fenics.Function | ufl.Form = (
            topological_derivative_pos
        )
        self.topological_derivative_neg: fenics.Function | ufl.Form = (
            topological_derivative_neg
        )
        self.update_levelset: Callable = update_levelset
        self.riesz_scalar_products = riesz_scalar_products
        self.volume_restriction = volume_restriction

        self.mesh = self.levelset_function.function_space().mesh()
        self.dg0_space = fenics.FunctionSpace(self.mesh, "DG", 0)

        self.cost_functional_form_deflation = _utils.enlist(
            self.cost_functional_form_initial
        )

    def _solve_inner_problem(
        self,
        tol: float = 1e-2,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
        angle_tol: Optional[float] = 1.0,
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

        self.levelset_function.vector().vec().aypx(
            0.0, self.levelset_function_init.vector().vec()
        )
        self.levelset_function.vector().apply("")

        topology_optimization_problem_inner = (
            topology_optimization.TopologyOptimizationProblem(
                self.state_forms,
                self.bcs_list,
                self.cost_functional_form_deflation,
                self.states,
                self.adjoints,
                self.levelset_function,
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.update_levelset,
                volume_restriction=self.volume_restriction,
                config=self.config,
                riesz_scalar_products=self.riesz_scalar_products,
                initial_guess=self.initial_guess,
                ksp_options=self.ksp_options,
                adjoint_ksp_options=self.adjoint_ksp_options,
                preconditioner_forms=self.preconditioner_forms,
            )
        )
        topology_optimization_problem_inner.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )

        if inner_atol is not None:
            atol = inner_atol
        else:
            atol = self.initial_norm * tol / 10.0

        topology_optimization_problem_inner.solve(
            rtol=self.rtol, atol=atol, angle_tol=angle_tol
        )

        topology_optimization_problem_inner.plot_shape()
        pp.show()

        topology_optimization_problem_temp = (
            topology_optimization.TopologyOptimizationProblem(
                self.state_forms,
                self.bcs_list,
                self.cost_functional_form_initial,
                self.states,
                self.adjoints,
                self.levelset_function,
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.update_levelset_deflation,
                self.volume_restriction,
                config=self.config,
                riesz_scalar_products=self.riesz_scalar_products,
                initial_guess=self.initial_guess,
                ksp_options=self.ksp_options,
                adjoint_ksp_options=self.adjoint_ksp_options,
                preconditioner_forms=self.preconditioner_forms,
            )
        )
        temp_problem.state_problem.has_solution = True
        self.current_function_value = temp_problem.reduced_cost_functional.evaluate()

    def solve(
        self,
        tol: float = 1e-2,
        it_deflation: int = 5,
        gamma: float = 0.5,
        delta: float = 1.0,
        inner_rtol: Optional[float] = None,
        inner_atol: Optional[float] = None,
        angle_tol: Optional[float] = 1.0,
    ) -> None:
        """Solves the constrained optimization problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            it_deflation: Number of performed deflation loops. Default is 5.
            gamma: Parameter to control the support of the penalty functions for the
                deflation procedure.
            delta: Penalty parameter of the penalty functions for the deflation
            procedure.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
            so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.

        """
        self.tol = tol

        self._solve_inner_problem(tol, inner_rtol, inner_atol, angle_tol)

        """characteristic_function = fenics.Function(self.dg0_space)
        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1., 0., characteristic_function
        )

        levelset_function_temp = fenics.Function(
            self.levelset_function.function_space()
        )
        levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function_.vector().vec()
        )
        levelset_function_temp.vector().apply("")

        characteristic_function_list = [characteristic_function]
        levelset_function_list = [levelset_function_temp]

        num_defl = 0
        while num_defl < it_deflation:

            characteristic_function_new = fenics.Function(self.dg0_space)

            for i in range(0, len(characteristic_function_list)):

                J_deflation = cost_functional.ScalarTrackingFunctional(
                    fenics.inner(
                        characteristic_function_new - characteristic_function_list[i],
                        characteristic_function_new - characteristic_function_list[i])
                )

                self.cost_functional_form_deflation.append(J_deflation)

            self._solve_inner_problem(tol, inner_rtol, inner_atol, angle_tol)

            characteristic_function = fenics.Function(self.dg0_space)
            _utils.interpolate_levelset_function_to_cells(
                self.levelset_function, 1., 0., characteristic_function
            )

            levelset_function_temp = fenics.Function(
                self.levelset_function.function_space()
            )
            levelset_function_temp.vector().vec().aypx(
                0.0, self.levelset_function_.vector().vec()
            )
            levelset_function_temp.vector().apply("")

            characteristic_function_list.append(characteristic_function)
            levelset_function_list.append(levelset_function_temp)

            self.cost_functional_form_deflation = _utils.enlist(
                self.cost_functional_form_initial
            )"""
