# Copyright (C) 2020-2025 Fraunhofer ITWM, Sebastian Blauth and
# Leon Baeck
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

from __future__ import annotations

import abc
import copy
import pathlib
from typing import Callable, TYPE_CHECKING

import fenics
from typing_extensions import Literal

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _utils
from cashocs import log
from cashocs._database import database
from cashocs._optimization import cost_functional
from cashocs._optimization import optimal_control
from cashocs._optimization import topology_optimization
from cashocs.io import output

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io


class DeflatedProblem(abc.ABC):
    """Base class for deflated problems for topology optimization."""

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
            ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
            adjoint_ksp_options: _typing.KspOption | list[
                _typing.KspOption] | None = None,
            gradient_ksp_options: _typing.KspOption | list[
                _typing.KspOption] | None = None,
            preconditioner_forms: list[ufl.Form] | None = None,
            pre_callback: (
                    Callable[[], None] | Callable[
                [_typing.OptimizationProblem], None] | None
            ) = None,
            post_callback: (
                    Callable[[], None] | Callable[
                [_typing.OptimizationProblem], None] | None
            ) = None,
            linear_solver: _utils.linalg.LinearSolver | None = None,
            adjoint_linear_solver: _utils.linalg.LinearSolver | None = None,
            newton_linearizations: ufl.Form | list[ufl.Form] | None = None,
            excluded_from_time_derivative: list[int] | list[list[int]] | None = None,
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
            constraint_list: (A list of) additional equality and inequality constraints
                for the problem.
            config: config: The config file for the problem, generated via
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

        """
        self.state_forms = _utils.enlist(state_forms)
        self.bcs_list = _utils.check_and_enlist_bcs(bcs_list)
        self.states = _utils.enlist(states)
        self.adjoints = _utils.enlist(adjoints)

        if config is not None:
            self.config = config
        else:
            self.config = io.Config()
        self.config.validate_config()

        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.gradient_ksp_options = gradient_ksp_options
        self.preconditioner_forms = preconditioner_forms
        self.pre_callback = pre_callback
        self.post_callback = post_callback
        self.linear_solver = linear_solver
        self.adjoint_linear_solver = adjoint_linear_solver
        self.newton_linearizations = newton_linearizations
        self.excluded_from_time_derivative = excluded_from_time_derivative

        self.current_function_value = 0.0

        self._pre_callback: (
                Callable[[], None] | Callable[
            [_typing.OptimizationProblem], None] | None
        ) = None
        self._post_callback: (
                Callable[[], None] | Callable[
            [_typing.OptimizationProblem], None] | None
        ) = None

        self.cost_functional_form_initial: list[_typing.CostFunctional] = _utils.enlist(
            cost_functional_form
        )

        self.db = database.Database(
            self.config,
            self.states,
            self.adjoints,
            self.ksp_options,  # type: ignore
            self.adjoint_ksp_options,  # type: ignore
            self.gradient_ksp_options,  # type: ignore
            self.cost_functional_form_initial,
            self.state_forms,
            self.bcs_list,
            self.preconditioner_forms,  # type: ignore
        )

        self.output_manager = output.OutputManager(self.db)

        self.abstract_control = fenics.Function
        self.abstract_control_init = fenics.function
        self.abstract_control_mapped = fenics.function
        self.control_list_mapped = []
        self.control_list_mapped_restart = []
        self.control_list_mapped_final = []
        self.control_list_final = []

    def distance_shapes(self):
        """Computes the distance of a new local minimizer of the deflated problem
           to all previously found ones.

                Returns:
                    True if the distance of the new local minimizer to all previous
                    found ones exceeds the threshold gamma. False if at least one
                    distance does not meet the threshold (no new minimizer of the
                    actual problem found).

                """
        for i in range(0, len(self.control_list_mapped_final) - 1):
            dist = fenics.assemble(
                fenics.inner(
                    self.control_list_mapped_final[i] -
                    self.control_list_mapped_restart[-1],
                    self.control_list_mapped_final[i] -
                    self.control_list_mapped_restart[-1]
                )
                * self.dx
            )

            if dist < 0.1 * self.gamma:
                return False

        return True

    @abc.abstractmethod
    def _solve_inner_problem(
            self,
            tol: float = 1e-2,
            inner_rtol: float | None = None,
            inner_atol: float | None = None,
            iteration: int = 0,
            angle_tol: Optional[float] = 1.0,
            restart: Optional[bool] = False,
    ) -> None:
        """Solves the inner (unpenalized) optimization problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            iteration: The current outer iteration count.
            angle_tol: The absolute tolerance for the angle between topological
                derivative and levelset function. If this is ``None``, then
                the value provided in the config file is used. Default is ``None``.
            restart: If True a restart of the unpenalized optimization problem with
                starting value as the minimizer of the perturbed optimization problem
                is initiated.


        """
        self.rtol = inner_rtol or tol

    def inject_pre_callback(
            self,
            function: (
                    Callable[[], None] | Callable[
                [_typing.OptimizationProblem], None] | None
            ),
    ) -> None:
        """Changes the a-priori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self._pre_callback = function

    def inject_post_callback(
            self,
            function: (
                    Callable[[], None] | Callable[
                [_typing.OptimizationProblem], None] | None
            ),
    ) -> None:
        """Changes the a-posteriori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self._post_callback = function

    def inject_pre_post_callback(
            self,
            pre_function: (
                    Callable[[], None] | Callable[
                [_typing.OptimizationProblem], None] | None
            ),
            post_function: (
                    Callable[[], None] | Callable[
                [_typing.OptimizationProblem], None] | None
            ),
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

    def check_for_restart(self):
        """Checks if a restart of the optimization of the unpenalized problem is needed.

                Returns:
                    Bool that indicates if a restart of the unpenalized problem with
                    the local minimizer of the deflated problem is needed. This is the
                    case if not all penality functions vanish in the local minimizer of
                    the deflated problem.

        """

        for i in range(1, len(self.cost_functional_form_deflation)):
            val = self.cost_functional_form_deflation[i].evaluate()
            if val > 1e-6:
                return True

        return False

    def reset_starting_value(self) -> None:
        """Resets the starting value for the optimization problem. """
        self.abstract_control.vector().vec().aypx(
            0.0, self.abstract_control_init.vector().vec()
        )
        self.abstract_control.vector().apply("")

    @abc.abstractmethod
    def construct_penalty_functions(
            self,
            gamma,
            delta
    ) -> None:
        """Constructs the penalty functions for the deflation procedure.

        Args:
            gamma: Threshold value for the local support of the penalty function. If the
                distance of two shapes is larger than this threshold, the penalty
                function vanishes.
            delta: Penalty parameter of the penalty function.

        """
        pass

    @abc.abstractmethod
    def map_abstract_control(self) -> fenics.Function:
        """Maps the abstract control for the optimization problem.

        Returns:
            abstract_control_mapped: A Fenics.function.

        """
        pass

    def save_functions(
            self,
            argument: Union[str, List[str]]
    ) -> None:
        """Stores the newly computed local minimizers during the deflation procedure.

        Args:
            Argument: str that indicates in which list the abstract_control of the 
                newly computed minimizer gets stored. 'solution': local minimizer of the
                deflated optimization problems. 'restart': local minimizers of the 
                restart procedure. 'final': (distinct) local minimizers of the 
                undeflated optimization problem.

        """
        arg_list = _utils.enlist(argument)

        abstract_control_temp = fenics.Function(
            self.abstract_control.function_space()
        )
        abstract_control_temp.vector().vec().aypx(
            0.0, self.abstract_control.vector().vec()
        )
        abstract_control_temp.vector().apply("")

        abstract_control_mapped_temp = self.map_abstract_control()

        if 'solution' in arg_list:
            self.control_list_mapped.append(abstract_control_mapped_temp)
        if 'restart' in arg_list:
            self.control_list_mapped_restart.append(abstract_control_mapped_temp)
        if 'final' in arg_list:
            self.control_list_mapped_final.append(abstract_control_mapped_temp)
            self.control_list_final.append(abstract_control_temp)

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
        """Deflation procedure to find multiple local minimizers of the optimization 
            problem.

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
            angle_tol: Absolute tolerance for the inner topology optimization problem.
                Default is ``None``.
        """
        self.gamma = gamma
        self.delta = delta
        
        self.reset_starting_value()

        log.begin("Begin of the deflation procedure.", level=log.INFO)
        log.info('Iteration 0 of Deflation loop:')

        self._solve_inner_problem(tol, inner_rtol, inner_atol, 0, angle_tol)
        self.save_functions(['solution', 'restart', 'final'])

        for num_defl in range(1, it_deflation + 1):

            log.info('Iteration {it} of Deflation loop:'.format(it=num_defl))
            
            self.cost_functional_form_deflation = \
                self.cost_functional_form_initial.copy()
            
            self.construct_penalty_functions(gamma, delta)

            self.reset_starting_value()
            self._solve_inner_problem(tol, inner_rtol, inner_atol, num_defl, angle_tol)
            self.save_functions('solution')

            restart = self.check_for_restart()

            self.cost_functional_form_deflation = \
                self.cost_functional_form_initial.copy()

            if restart:
                log.info('Performing a Restart (At least one penalty function did '
                         'not vanish)')
                self._solve_inner_problem(
                    tol,
                    inner_rtol,
                    inner_atol,
                    num_defl,
                    angle_tol,
                    restart=True
                )

            self.save_functions('restart')
            distance = self.distance_shapes()

            if distance:
                log.info('New local Minimizer found')
                self.save_functions('final')
            else:
                log.info('Minimizer was already computed before')

        log.end()


class DeflatedTopologyOptimizationProblem(DeflatedProblem):
    """A deflated topology optimization problem."""

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
            ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
            adjoint_ksp_options: _typing.KspOption | list[
                _typing.KspOption] | None = None,
            gradient_ksp_options: _typing.KspOption | list[
                _typing.KspOption] | None = None,
            preconditioner_forms: list[ufl.Form] | ufl.Form | None = None,
            pre_callback: Callable | None = None,
            post_callback: Callable | None = None,
            linear_solver: _utils.linalg.LinearSolver | None = None,
            adjoint_linear_solver: _utils.linalg.LinearSolver | None = None,
            newton_linearizations: ufl.Form | list[ufl.Form] | None = None,
            excluded_from_time_derivative: list[int] | list[list[int]] | None = None,
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
            levelset_function: A :py:class:`fenics.Function` which represents the
                levelset function.
            topological_derivative_neg: The topological derivative inside the domain,
                where the levelset function is negative.
            topological_derivative_pos: The topological derivative inside the domain,
                where the levelset function is positive.
            update_levelset: A python function (without arguments) which is called to
                update the coefficients etc. when the levelset function is changed.
            volume_restriction: A volume restriction for the optimization problem.
                A single float describes an equality constraint and a tuple of floats
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
            initial_guess: list of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).
            gradient_ksp_options: A list of dicts corresponding to command line options
                for PETSc, used to compute the (shape) gradient. If this is ``None``,
                either a direct or an iterative method is used (depending on the
                configuration, section OptimizationRoutine, key gradient_method).
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
            preconditioner_forms=preconditioner_forms,
            pre_callback=pre_callback,
            post_callback=post_callback,
            linear_solver=linear_solver,
            adjoint_linear_solver=adjoint_linear_solver,
            newton_linearizations=newton_linearizations,
            excluded_from_time_derivative=excluded_from_time_derivative,
        )

        self.abstract_control: fenics.Function = levelset_function
        self.abstract_control_init = fenics.Function(
            self.abstract_control.function_space()
        )
        self.abstract_control_init.vector().vec().aypx(
            0.0, self.abstract_control.vector().vec()
        )
        self.abstract_control_init.vector().apply("")

        self.topological_derivative_pos: fenics.Function | ufl.Form = (
            topological_derivative_pos
        )
        self.topological_derivative_neg: fenics.Function | ufl.Form = (
            topological_derivative_neg
        )
        self.update_levelset: Callable = update_levelset
        self.riesz_scalar_products = riesz_scalar_products
        self.volume_restriction = volume_restriction

        self.db.function_db.controls = _utils.enlist(self.abstract_control)
        self.db.function_db.control_spaces = [
            x.function_space() for x in self.db.function_db.controls
        ]
        self.db.parameter_db.problem_type = "topology"

        self.mesh = self.abstract_control.function_space().mesh()
        self.dg0_space = fenics.FunctionSpace(self.mesh, "DG", 0)

        self.cost_functional_form_deflation = self.cost_functional_form_initial.copy()

        self.dx = fenics.Measure("dx", self.abstract_control.function_space().mesh())
        self.abstract_control_mapped = fenics.Function(self.dg0_space)
        
    def construct_penalty_functions(self, gamma, delta) -> None:
        """Constructs the penalty functions for the deflation procedure.

        Args:
            gamma: Threshold value for the local support of the penalty function. If the
                distance of two shapes is larger than this threshold, the penalty
                function vanishes.
            delta: Penalty parameter of the penalty function.

        """
        self.abstract_control_mapped = fenics.Function(self.dg0_space)

        for i in range(0, len(self.control_list_mapped)):
            J_deflation = cost_functional.DeflationFunctional(
                gamma,
                fenics.inner(
                    self.abstract_control_mapped -
                    self.control_list_mapped[i],
                    self.abstract_control_mapped -
                    self.control_list_mapped[i])
                * self.dx,
                (1 - 2 * self.control_list_mapped[i]),
                delta
            )

            self.cost_functional_form_deflation.append(J_deflation)
            
            
    def map_abstract_control(self) -> fenics.Function:
        """Maps the abstract control for the optimization problem.

        Returns:
            abstract_control_mapped: The mapped abstract control is defined as the
                characteristic function of the domain that is given by the level-set
                function (abstract_control).

        """
        abstract_control_mapped_temp = fenics.Function(self.dg0_space)
        _utils.interpolate_levelset_function_to_cells(
            self.abstract_control, 1., 0., abstract_control_mapped_temp
        )
        return abstract_control_mapped_temp

    def update_level_set_deflation(self):
        """update_levelset function for the deflated topology optimization problem."""
        self.update_levelset()
        _utils.interpolate_levelset_function_to_cells(
            self.abstract_control, 1., 0., self.abstract_control_mapped
        )

    def _solve_inner_problem(
            self,
            tol: float = 1e-2,
            inner_rtol: Optional[float] = None,
            inner_atol: Optional[float] = None,
            iteration: int = 0,
            angle_tol: Optional[float] = 1.0,
            restart: Optional[bool] = False,
    ) -> None:
        """Solves the inner (unpenalized) optimization problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            iteration: The current outer iteration count
            angle_tol: The absolute tolerance for the angle between topological
                derivative and levelset function. If this is ``None``, then
                the value provided in the config file is used. Default is ``None``.
            restart: If True a restart of the unpenalized optimization problem with
                starting value as the minimizer of the perturbed optimization problem
                is initiated.

        """
        super()._solve_inner_problem(
            tol=tol, inner_rtol=inner_rtol, inner_atol=inner_atol, iteration=iteration
        )

        config = copy.deepcopy(self.config)
        output_path = pathlib.Path(self.config.get("Output", "result_dir"))
        if restart:
            config.set("Output", "result_dir",
                       str(output_path / f"subproblem_{iteration}_restart"))
        else:
            config.set("Output", "result_dir",
                       str(output_path / f"subproblem_{iteration}"))

        topology_optimization_problem_inner = (
            topology_optimization.TopologyOptimizationProblem(
                self.state_forms,
                self.bcs_list,
                self.cost_functional_form_deflation,
                self.states,
                self.adjoints,
                self.abstract_control,
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.update_level_set_deflation,
                volume_restriction=self.volume_restriction,
                config=config,
                riesz_scalar_products=self.riesz_scalar_products,
                initial_guess=self.initial_guess,
                ksp_options=self.ksp_options,
                adjoint_ksp_options=self.adjoint_ksp_options,
                gradient_ksp_options=self.gradient_ksp_options,
                preconditioner_forms=self.preconditioner_forms,
                pre_callback=self.pre_callback,
                post_callback=self.post_callback,
                linear_solver=self.linear_solver,
                adjoint_linear_solver=self.adjoint_linear_solver,
                newton_linearizations=self.newton_linearizations,
                excluded_from_time_derivative=self.excluded_from_time_derivative,
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


class DeflatedOptimalControlProblem(DeflatedProblem):
    """A deflated optimal control problem."""

    def __init__(
            self,
            state_forms: ufl.Form | list[ufl.Form],
            bcs_list: (
                    fenics.DirichletBC
                    | list[fenics.DirichletBC]
                    | list[list[fenics.DirichletBC]]
            ),
            cost_functional_form: list[_typing.CostFunctional] | _typing.CostFunctional,
            states: fenics.Function | list[fenics.Function],
            controls: fenics.Function | list[fenics.Function],
            adjoints: fenics.Function | list[fenics.Function],
            config: io.Config | None = None,
            riesz_scalar_products: ufl.Form | list[ufl.Form] | None = None,
            control_constraints: list[list[float | fenics.Function]] | None = None,
            initial_guess: list[fenics.Function] | None = None,
            ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
            adjoint_ksp_options: _typing.KspOption | list[
                _typing.KspOption] | None = None,
            gradient_ksp_options: _typing.KspOption | list[
                _typing.KspOption] | None = None,
            control_bcs_list: (
                    fenics.DirichletBC
                    | list[fenics.DirichletBC]
                    | list[list[fenics.DirichletBC]]
                    | None
            ) = None,
            preconditioner_forms: list[ufl.Form] | None = None,
            pre_callback: Callable | None = None,
            post_callback: Callable | None = None,
            linear_solver: _utils.linalg.LinearSolver | None = None,
            adjoint_linear_solver: _utils.linalg.LinearSolver | None = None,
            newton_linearizations: ufl.Form | list[ufl.Form] | None = None,
            excluded_from_time_derivative: list[int] | list[list[int]] | None = None,
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
            preconditioner_forms=preconditioner_forms,
            pre_callback=pre_callback,
            post_callback=post_callback,
            linear_solver=linear_solver,
            adjoint_linear_solver=adjoint_linear_solver,
            newton_linearizations=newton_linearizations,
            excluded_from_time_derivative=excluded_from_time_derivative,
        )

        self.controls = _utils.enlist(controls)
        self.abstract_control = self.controls[0]
        self.abstract_control_mapped = self.abstract_control
        self.riesz_scalar_products = riesz_scalar_products
        self.control_bcs_list = control_bcs_list
        self.control_constraints = control_constraints

        self.db.function_db.controls = self.controls
        self.db.function_db.control_spaces = [
            x.function_space() for x in self.db.function_db.controls
        ]
        self.db.function_db.gradient = _utils.create_function_list(
            self.db.function_db.control_spaces
        )
        self.db.parameter_db.problem_type = "control"

        self.abstract_control_init = fenics.Function(
            self.abstract_control.function_space()
        )
        self.abstract_control_init.vector().vec().aypx(
            0.0, self.abstract_control.vector().vec()
        )
        self.abstract_control_init.vector().apply("")

        self.cost_functional_form_deflation = self.cost_functional_form_initial.copy()

        self.dx = fenics.Measure("dx", self.abstract_control.function_space().mesh())
        
    def construct_penalty_functions(self, gamma, delta) -> None:
        """Constructs the penalty functions for the deflation procedure.

        Args:
            gamma: Threshold value for the local support of the penalty function. If the
                distance of two shapes is larger than this threshold, the penalty
                function vanishes.
            delta: Penalty parameter of the penalty function.

        """

        for i in range(0, len(self.control_list_mapped)):
            J_deflation = cost_functional.DeflationFunctional(
                self.gamma,
                fenics.inner(
                    self.abstract_control - self.control_list_mapped[i],
                    self.abstract_control - self.control_list_mapped[i])
                * self.dx,
                ufl.Form,
                self.delta
            )

            self.cost_functional_form_deflation.append(J_deflation)
            
    def map_abstract_control(self) -> fenics.Function:
        """Maps the abstract control for the optimization problem.

        Returns:
            abstract_control_mapped: The mapped abstract control coincides with the
                abstract control.

        """
        abstract_control_mapped_temp = fenics.Function(
            self.abstract_control.function_space()
        )
        abstract_control_mapped_temp.vector().vec().aypx(
            0.0, self.abstract_control.vector().vec()
        )
        abstract_control_mapped_temp.vector().apply("")
        return abstract_control_mapped_temp

    def _solve_inner_problem(
            self,
            tol: float = 1e-2,
            inner_rtol: float | None = None,
            inner_atol: float | None = None,
            iteration: int = 0,
            angle_tol: Optional[float] = 1.0,
            restart: Optional[bool] = False,
    ) -> None:
        """Solves the inner (unpenalized) optimization problem.

        Args:
            tol: An overall tolerance to be used in the algorithm. This will set the
                relative tolerance for the inner optimization problems to ``tol``.
                Default is 1e-2.
            inner_rtol: Relative tolerance for the inner problem. Default is ``None``,
                so that ``inner_rtol = tol`` is used.
            inner_atol: Absolute tolerance for the inner problem. Default is ``None``,
                so that ``inner_atol = tol/10`` is used.
            iteration: The current outer iteration count
            restart: If True a restart of the unpenalized optimization problem with
                starting value as the minimizer of the perturbed optimization problem
                is initiated.

        """
        super()._solve_inner_problem(
            tol=tol,
            inner_rtol=inner_rtol,
            inner_atol=inner_atol,
            iteration=iteration
        )

        config = copy.deepcopy(self.config)
        output_path = pathlib.Path(self.config.get("Output", "result_dir"))
        if restart:
            config.set(
                "Output",
                "result_dir",
                str(output_path / f"subproblem_{iteration}_restart")
            )
        else:
            config.set(
                "Output",
                "result_dir",
                str(output_path / f"subproblem_{iteration}")
            )

        optimal_control_problem = optimal_control.OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form_deflation,
            self.states,
            self.controls,
            self.adjoints,
            config=config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            gradient_ksp_options=self.gradient_ksp_options,
            control_bcs_list=self.control_bcs_list,
            preconditioner_forms=self.preconditioner_forms,
            pre_callback=self.pre_callback,
            post_callback=self.post_callback,
            linear_solver=self.linear_solver,
            adjoint_linear_solver=self.adjoint_linear_solver,
            newton_linearizations=self.newton_linearizations,
            excluded_from_time_derivative=self.excluded_from_time_derivative,
        )

        optimal_control_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )

        if inner_atol is not None:
            atol = inner_atol
        else:
            atol = self.initial_norm * tol / 10.0

        optimal_control_problem.solve(rtol=self.rtol, atol=atol)
