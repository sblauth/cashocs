# Copyright (C) 2020-2024 Sebastian Blauth
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

"""Space mapping for topology optimization problems."""

from __future__ import annotations

import abc
import collections
import json
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
from typing_extensions import Literal

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs._optimization.topology_optimization import topology_optimization_problem as top
from cashocs._optimization.topology_optimization import bisection

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io


class FineModel(abc.ABC):
    """Base class for the fine model in space mapping topology optimization.

    Attributes:
        mesh: The FEM mesh for the fine model.
        cost_functional_value: The current cost functional value of the fine model.

    """

    cost_functional_value: float

    def __init__(self, mesh: fenics.Mesh):
        """Initializes self.

        Args:
            mesh: The finite element mesh of the coarse model, used for the space
                mapping with the fine model.

        """
        self.mesh = fenics.Mesh(mesh)

    @abc.abstractmethod
    def solve_and_evaluate(self) -> None:
        """Solves and evaluates the fine model.

        This needs to be overwritten with a custom implementation.

        """
        pass


class CoarseModel:
    """Coarse Model for space mapping topology optimization."""

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
        desired_weights: list[float] | None = None,
        preconditioner_forms: Optional[Union[List[ufl.Form], ufl.Form]] = None,
    ) -> None:

        r"""Initializes the coarse model topology optimization problem.

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
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.

        """
        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.cost_functional_form = cost_functional_form
        self.states = states
        self.adjoints = adjoints
        self.levelset_function = levelset_function
        self.topological_derivative_neg = topological_derivative_neg
        self.topological_derivative_pos = topological_derivative_pos
        self.update_levelset = update_levelset
        self.volume_restriction = volume_restriction
        self.config = config
        self.riesz_scalar_products = riesz_scalar_products
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.desired_weights = desired_weights
        self.preconditioner_forms = preconditioner_forms

        self._pre_callback: Optional[Callable] = None
        self._post_callback: Optional[Callable] = None

        self.mesh = self.levelset_function.function_space().mesh()
        self.levelset_function_initial = fenics.Function(
            self.levelset_function.function_space()
        )
        self.levelset_function_initial.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_initial.vector().apply("")

        self.levelset_function_optimal = fenics.Function(
            self.levelset_function.function_space()
        )

        self.topology_optimization_problem =  top.TopologyOptimizationProblem(
                self.state_forms,
                self.bcs_list,
                self.cost_functional_form,
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
                desired_weights=self.desired_weights,
                preconditioner_forms=self.preconditioner_forms,
        )

    def optimize(self) -> None:
        """Solves the coarse model optimization problem."""
        self.topology_optimization_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )
        self.topology_optimization_problem.solve()

        self.levelset_function_optimal.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_optimal.vector().apply("")
        
        fenics.plot(self.levelset_function_optimal, vmin=-1e-10, vmax=1e-10)
        from matplotlib import  pyplot as plt
        plt.show()


class ParameterExtraction:
    """Parameter extraction for space mapping topology optimization."""
    
    def __init__(
        self,
        coarse_model: CoarseModel,
        cost_functional_form: list[_typing.CostFunctional] | _typing.CostFunctional,
        states: list[fenics.Function] | fenics.Function,
        adjoints: list[fenics.Function] | fenics.Function,
        topological_derivative_neg: fenics.Function | ufl.Form,
        topological_derivative_pos: fenics.Function | ufl.Form,
        config: io.Config | None = None,
        desired_weights: list[float] | None = None,
        mode: str = "initial",
    ) -> None:

        r"""Initializes the parameter extraction topology optimization problem.

        Args:
            coarse_model: The coarse model optimization problem
            cost_functional_form: The cost functional for the parameter extraction
            states: The state variables for the parameter extraction
            adjoints: The adjoint variables for the parameter extraction
            topological_derivative_neg: The topological derivative inside the domain,
                where the levelset function is negative.
            topological_derivative_pos: The topological derivative inside the domain,
                where the levelset function is positive.
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            desired_weights: The list of desired weights for the parameter extraction
            mode: The mode used for the initial guess of the parameter extraction. If
                this is coarse_optimum, the default, then the coarse model optimum is
                used as initial guess, if this is initial, then the initial guess for
                the optimization is used.

        """
        self.coarse_model = coarse_model
        self.mesh = coarse_model.mesh
        self.cost_functional_form = cost_functional_form
        self.topological_derivative_neg = topological_derivative_neg
        self.topological_derivative_pos = topological_derivative_pos
        self.mode = mode

        self.states = _utils.enlist(states)
        self.adjoints = _utils.enlist(adjoints)

        self.config = config
        self.desired_weights = desired_weights

        self._pre_callback: Optional[Callable] = None
        self._post_callback: Optional[Callable] = None

        dict_states = {
            coarse_model.topology_optimization_problem.states[i]: self.states[i]
            for i in range(len(self.states))
        }
        dict_adjoints = {
            coarse_model.topology_optimization_problem.adjoints[i]: self.adjoints[i]
            for i in range(len(self.adjoints))
        }
        mapping_dict = {}
        mapping_dict.update(dict_states)
        mapping_dict.update(dict_adjoints)

        self.state_forms = [
            ufl.replace(form, mapping_dict)
            for form in coarse_model.topology_optimization_problem.state_forms
        ]
        self.bcs_list = coarse_model.topology_optimization_problem.bcs_list

        self.levelset_function = (
            coarse_model.topology_optimization_problem.levelset_function
        )
        self.update_levelset = (
            coarse_model.topology_optimization_problem.update_levelset
        )
        self.volume_restriction = coarse_model.volume_restriction
        self.riesz_scalar_products = (
            coarse_model.topology_optimization_problem.riesz_scalar_products
        )
        self.initial_guess = coarse_model.topology_optimization_problem.initial_guess
        self.ksp_options = coarse_model.topology_optimization_problem.ksp_options
        self.adjoint_ksp_options = (
            coarse_model.topology_optimization_problem.adjoint_ksp_options
        )
        self.preconditioner_forms = (
            coarse_model.topology_optimization_problem.preconditioner_forms
        )


        self.levelset_function_optimal = fenics.Function(
            coarse_model.levelset_function.function_space()
        )
        self.levelset_function_optimal.vector().vec().aypx(
            0.0, coarse_model.levelset_function_optimal.vector().vec()
        )
        self.levelset_function_optimal.vector().apply("")

        self.levelset_function_initial = fenics.Function(
            coarse_model.levelset_function.function_space()
        )
        self.levelset_function_initial.vector().vec().aypx(
            0.0, coarse_model.levelset_function_initial.vector().vec()
        )
        self.levelset_function_initial.vector().apply("")

        self.topology_optimization_problem: Optional[
            top.TopologyOptimizationProblem] = None

    def _solve(self) -> None:
        """Solves the parameter extraction problem.

        Args:
            initial_guess: The initial guesses for solving the problem.

        """
        if self.mode == "initial":
            self.levelset_function.vector().vec().aypx(
                0.0, self.levelset_function_initial.vector().vec()
            )
            self.levelset_function.vector().apply("")
        elif self.mode == "coarse_optimum":
            self.levelset_function.vector().vec().aypx(
                0.0, self.levelset_function_optimal.vector().vec()
            )
            self.levelset_function.vector().apply("")

        self.topology_optimization_problem = top.TopologyOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
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
            desired_weights=self.desired_weights,
            preconditioner_forms=self.preconditioner_forms,
        )
        if self.topology_optimization_problem is not None:
            self.topology_optimization_problem.inject_pre_post_callback(
                self._pre_callback, self._post_callback
            )
            
            self.topology_optimization_problem.solve()
            

class SpaceMappingProblem:
    """Space mapping method for topology optimization."""

    def __init__(
        self,
        fine_model: FineModel,
        coarse_model: CoarseModel,
        parameter_extraction: ParameterExtraction,
        method: Literal[
            "broyden", "bfgs", "lbfgs", "sd", "cc", "ncg"
        ] = "broyden",
        max_iter: int = 25,
        tol: float = 1e-2,
        use_backtracking_line_search: bool = False,
        broyden_type: Literal["good", "bad"] = "good",
        cg_type: Literal["FR", "PR", "HS", "DY", "HZ"] = "FR",
        memory_size: int = 10,
        verbose: bool = True,
        save_history: bool = False,
    ) -> None:
        """Initializes self.

        Args:
            fine_model: The fine model optimization problem
            coarse_model: The coarse model optimization problem
            parameter_extraction: The parameter extraction problem
            method: A string, which indicates which method is used to solve the space
                mapping. Can be one of "broyden", "bfgs", "lbfgs", "sd",
                "steepest descent", or "ncg". Default is "broyden".
            max_iter: Maximum number of space mapping iterations
            tol: The tolerance used for solving the space mapping iteration
            use_backtracking_line_search: A boolean flag, which indicates whether a
                backtracking line search should be used for the space mapping.
            broyden_type: A string, either "good" or "bad", determining the type of
                Broyden's method used. Default is "good"
            cg_type: A string, either "FR", "PR", "HS", "DY", "HZ", which indicates
                which NCG variant is used for solving the space mapping. Default is "FR"
            memory_size: The size of the memory for Broyden's method and the BFGS method
            verbose: A boolean flag which indicates, whether the output of the space
                mapping method should be verbose. Default is ``True``.
            save_history: A boolean flag which indicates, whether the history of the
                space mapping method should be saved to a .json file. Default is
                ``False``.

        """
        self.fine_model = fine_model
        self.coarse_model = coarse_model
        self.parameter_extraction = parameter_extraction
        self.method = method
        if self.method == "sd":
            self.method = "steepest_descent"
        elif self.method == "lbfgs":
            self.method = "bfgs"
        self.max_iter = max_iter
        self.tol = tol
        self.use_backtracking_line_search = use_backtracking_line_search
        self.broyden_type = broyden_type
        self.cg_type = cg_type
        self.memory_size = memory_size
        self.verbose = verbose
        self.save_history = save_history

        self.levelset_function_initial = fenics.Function(
            coarse_model.levelset_function.function_space()
        )
        self.levelset_function_initial.vector().vec().aypx(
            0.0, coarse_model.levelset_function_initial.vector().vec()
        )
        self.levelset_function_initial.vector().apply("")

        self.eps = 1.0
        self.converged = False
        self.iteration = 0

        self.x: fenics.Function = self.fine_model.levelset_function

        self.projection = bisection.LevelSetVolumeProjector(
            self.x, self.coarse_model.volume_restriction,
            self.coarse_model.topology_optimization_problem.db
        )

        self.levelset_space_fine = self.x.function_space()
        self.levelset_space_coarse = coarse_model.levelset_function.function_space()

        self.ips_to_coarse = _utils.Interpolator(
            self.levelset_space_fine, self.levelset_space_coarse
        )
        self.ips_to_fine = _utils.Interpolator(
            self.levelset_space_coarse, self.levelset_space_fine
        )

        self.z_star = [fenics.Function(self.levelset_space_coarse)]
        self.norm_z_star = 1.0
        self.p_current = [fenics.Function(self.levelset_space_coarse)]
        self.p_prev = [fenics.Function(self.levelset_space_coarse)]
        self.h = [fenics.Function(self.levelset_space_coarse)]
        self.v = [fenics.Function(self.levelset_space_coarse)]
        self.u = [fenics.Function(self.levelset_space_coarse)]
        self.transformation = fenics.Function(self.levelset_space_coarse)

        self.stepsize = 1.0
        self.x_save = fenics.Function(self.levelset_space_fine)

        self.diff = [fenics.Function(self.levelset_space_coarse)]
        self.temp = [fenics.Function(self.levelset_space_coarse)]
        self.dir_prev = [fenics.Function(self.levelset_space_coarse)]
        self.difference = [fenics.Function(self.levelset_space_coarse)]

        self.history_s: collections.deque = collections.deque()
        self.history_y: collections.deque = collections.deque()
        self.history_rho: collections.deque = collections.deque()
        self.history_alpha: collections.deque = collections.deque()

        self.space_mapping_history: Dict[str, List[float]] = {
            "cost_function_value": [],
            "eps": [],
            "stepsize": [],
        }

    def update_history(self) -> None:
        """Updates the space mapping history."""
        self.space_mapping_history["cost_function_value"].append(
            self.fine_model.cost_functional_value
        )
        self.space_mapping_history["eps"].append(self.eps)
        self.space_mapping_history["stepsize"].append(self.stepsize)

    def test_for_nonconvergence(self) -> None:
        """Tests, whether maximum number of iterations are exceeded."""
        if self.iteration >= self.max_iter:
            raise _exceptions.NotConvergedError(
                "Space Mapping",
                "Maximum number of iterations exceeded.",
            )

    def _compute_initial_guess(self) -> None:
        """Compute initial guess for the space mapping by solving the coarse problem."""
        self.coarse_model.optimize()
        self.z_star[0].vector().vec().aypx(
            0.0,
            self.coarse_model.levelset_function_optimal.vector().vec() -
            self.levelset_function_initial.vector().vec()
        )
        self.z_star[0].vector().apply("")
        
        self.x.vector().vec().aypx(
            0.0,
            self.ips_to_fine.interpolate(
                self.coarse_model.levelset_function_optimal
            ).vector().vec()
        )
        self.x.vector().apply("")

        self.norm_z_star = np.sqrt(self._scalar_product(self.z_star, self.z_star))

    def solve(self) -> None:
        """Solves the problem with the space mapping method."""
        self._compute_initial_guess()

        self.fine_model.solve_and_evaluate()
        self.parameter_extraction._solve()  # pylint: disable=protected-access

        self.p_current[0].vector().vec().aypx(
            0.0,
            self.parameter_extraction.levelset_function.vector().vec() -
            self.levelset_function_initial.vector().vec()
        )
        self.p_current[0].vector().apply("")
        self.eps = self._compute_eps()

        self.update_history()
        if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            print(
                f"Space Mapping - Iteration {self.iteration:3d}:"
                f"    Cost functional value = "
                f"{self.fine_model.cost_functional_value:.3e}"
                f"    eps = {self.eps:.3e}",
                flush=True,
            )
        fenics.MPI.barrier(fenics.MPI.comm_world)

        while not self.converged:
            self.dir_prev[0].vector().vec().aypx(
                0.0, -(self.p_prev[0].vector().vec() - self.z_star[0].vector().vec())
            )
            self.dir_prev[0].vector().apply("")
            self.temp[0].vector().vec().aypx(
                0.0, -(self.p_current[0].vector().vec() - self.z_star[0].vector().vec())
            )
            self.temp[0].vector().apply("")
            self._compute_search_direction(self.temp, self.h)

            self.stepsize = 1.0
            self.p_prev[0].vector().vec().aypx(0.0, self.p_current[0].vector().vec())
            self.p_prev[0].vector().apply("")
            self._update_iterates()

            self.iteration += 1

            self.update_history()
            if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                print(
                    f"Space Mapping - Iteration {self.iteration:3d}:"
                    f"    Cost functional value = "
                    f"{self.fine_model.cost_functional_value:.3e}"
                    f"    eps = {self.eps:.3e}"
                    f"    step size = {self.stepsize:.3e}",
                    flush=True,
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)

            if self.eps <= self.tol:
                self.converged = True
                break
            self.test_for_nonconvergence()

            self._update_broyden_approximation()
            self._update_bfgs_approximation()

        if self.converged:
            if self.save_history and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                with open("./sm_history.json", "w", encoding="utf-8") as file:
                    json.dump(self.space_mapping_history, file, indent=4)
            fenics.MPI.barrier(fenics.MPI.comm_world)
            output = (
                f"\nStatistics --- "
                f"Space mapping iterations: {self.iteration:4d} --- "
                f"Final objective value: {self.fine_model.cost_functional_value:.3e}\n"
            )
            if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                print(output, flush=True)
            fenics.MPI.barrier(fenics.MPI.comm_world)

    def _update_broyden_approximation(self) -> None:
        """Updates the approximation of the mapping function with Broyden's method."""
        if self.method == "broyden":
            self.temp[0].vector().vec().aypx(
                0.0, self.p_current[0].vector().vec() - self.p_prev[0].vector().vec()
            )
            self.temp[0].vector().apply("")
            self._compute_broyden_application(self.temp, self.v)

            if self.memory_size > 0:
                if self.broyden_type == "good":
                    divisor = self._scalar_product(self.h, self.v)
                    self.u[0].vector().vec().axpby(
                        1.0 / divisor,
                        0.0,
                        self.h[0].vector().vec() - self.v[0].vector().vec(),
                    )
                    self.u[0].vector().apply("")

                    self.history_s.append([xx.copy(True) for xx in self.u])
                    self.history_y.append([xx.copy(True) for xx in self.h])

                elif self.broyden_type == "bad":
                    divisor = self._scalar_product(self.temp, self.temp)
                    self.u[0].vector().vec().axpby(
                        1.0 / divisor,
                        0.0,
                        self.h[0].vector().vec() - self.v[0].vector().vec(),
                    )
                    self.u[0].vector().apply("")

                    self.history_s.append([xx.copy(True) for xx in self.u])
                    self.history_y.append([xx.copy(True) for xx in self.temp])

                if len(self.history_s) > self.memory_size:
                    self.history_s.popleft()
                    self.history_y.popleft()

    def _update_bfgs_approximation(self) -> None:
        """Updates the approximation of the mapping function with the BFGS method."""
        if self.method == "bfgs":
            if self.memory_size > 0:
                self.temp[0].vector().vec().aypx(
                    0.0,
                    self.p_current[0].vector().vec() - self.p_prev[0].vector().vec(),
                )
                self.temp[0].vector().apply("")

                self.history_y.appendleft([xx.copy(True) for xx in self.temp])
                self.history_s.appendleft([xx.copy(True) for xx in self.h])
                curvature_condition = self._scalar_product(self.temp, self.h)

                if curvature_condition <= 0.0:
                    self.history_s.clear()
                    self.history_y.clear()
                    self.history_rho.clear()
                else:
                    rho = 1 / curvature_condition
                    self.history_rho.appendleft(rho)

                if len(self.history_s) > self.memory_size:
                    self.history_s.pop()
                    self.history_y.pop()
                    self.history_rho.pop()

    def _update_iterates(self) -> None:
        """Updates the iterates either directly or via a line search."""
        if not self.use_backtracking_line_search:
            
            self.x.vector().vec().axpy(
                1.0, self.ips_to_fine.interpolate(self.h[0]).vector().vec()
            )
            self.x.vector().apply("")
            self.projection.project()

            self.fine_model.solve_and_evaluate()
            self.parameter_extraction._solve()  # pylint: disable=protected-access
            self.p_current[0].vector().vec().aypx(
                0.0,
                self.parameter_extraction.levelset_function.vector().vec() -
                self.levelset_function_initial.vector().vec()
            )
            self.p_current[0].vector().apply("")
            self.eps = self._compute_eps()

        else:
            self.x_save.vector().vec().aypx(0.0, self.x.vector().vec())
            self.x_save.vector().apply("")

            while True:
                if self.stepsize <= 1e-4:
                    raise _exceptions.NotConvergedError(
                        "Space Mapping Backtracking Line Search",
                        "The line search did not converge.",
                    )

                self.x.vector().vec().aypx(0.0, self.x_save.vector().vec())
                self.x.vector().apply("")

                self.transformation.vector().vec().axpby(
                    self.stepsize, 0.0, self.h[0].vector().vec()
                )
                self.transformation.vector().apply("")

                self.x.vector().vec().axpy(
                    1.0, 
                    self.ips_to_fine.interpolate(self.transformation).vector().vec()
                )
                self.x.vector().apply("")
                self.projection.project()

                self.fine_model.solve_and_evaluate()
                # pylint: disable=protected-access
                self.parameter_extraction._solve()
                self.p_current[0].vector().vec().aypx(
                    0.0,
                    self.parameter_extraction.levelset_function.vector().vec() -
                    self.levelset_function_initial.vector().vec()
                )
                self.p_current[0].vector().apply("")
                eps_new = self._compute_eps()

                if eps_new <= self.eps:
                    self.eps = eps_new
                    break
                else:
                    self.stepsize /= 2

    def _scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between ``a`` and ``b``.

        Args:
            a: The first input for the scalar product
            b: The second input for the scalar product

        Returns:
            The scalar product between ``a`` and ``b``

        """
        return float(
            self.coarse_model.topology_optimization_problem.form_handler.scalar_product(
                a, b
            )
        )

    def _compute_search_direction(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for a given rhs ``q``, saved to ``out``.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.method == "steepest_descent":
            return self._compute_steepest_descent_application(q, out)
        elif self.method == "broyden":
            return self._compute_broyden_application(q, out)
        elif self.method == "bfgs":
            return self._compute_bfgs_application(q, out)
        elif self.method == "ncg":
            return self._compute_ncg_direction(q, out)
        else:
            raise _exceptions.InputError(
                "cashocs.space_mapping.shape_optimization.SpaceMapping",
                "method",
                "The method is not supported.",
            )

    @staticmethod
    def _compute_steepest_descent_application(
        q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for the steepest descent method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        for i in range(len(out)):
            out[i].vector().vec().aypx(0.0, q[i].vector().vec())
            out[i].vector().apply("")

    def _compute_broyden_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for Broyden's method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        out[0].vector().vec().aypx(0.0, q[0].vector().vec())
        out[0].vector().apply("")

        for i in range(len(self.history_s)):
            if self.broyden_type == "good":
                alpha = self._scalar_product(self.history_y[i], out)
            elif self.broyden_type == "bad":
                alpha = self._scalar_product(self.history_y[i], q)
            else:
                raise _exceptions.CashocsException(
                    "Type of Broyden's method has to be either 'good' or 'bad'."
                )

            out[0].vector().vec().axpy(alpha, self.history_s[i][0].vector().vec())
            out[0].vector().apply("")

    def _compute_bfgs_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for the LBFGS method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.memory_size > 0 and len(self.history_s) > 0:
            self.history_alpha.clear()
            out[0].vector().vec().aypx(0.0, q[0].vector().vec())
            out[0].vector().apply("")

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self._scalar_product(
                    self.history_s[i], out
                )
                self.history_alpha.append(alpha)
                out[0].vector().vec().axpy(-alpha, self.history_y[i][0].vector().vec())
                out[0].vector().apply("")

            bfgs_factor = self._scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self._scalar_product(self.history_y[0], self.history_y[0])
            out[0].vector().vec().scale(bfgs_factor)
            out[0].vector().apply("")

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self._scalar_product(
                    self.history_y[-1 - i], out
                )
                out[0].vector().vec().axpy(
                    self.history_alpha[-1 - i] - beta,
                    self.history_s[-1 - i][0].vector().vec(),
                )
                out[0].vector().apply("")

        else:
            out[0].vector().vec().aypx(0.0, q[0].vector().vec())
            out[0].vector().apply("")

    def _compute_ncg_direction(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for the NCG methods.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.iteration > 0:
            self.difference[0].vector().vec().aypx(
                0.0, q[0].vector().vec() - self.dir_prev[0].vector().vec()
            )
            self.difference[0].vector().apply("")

            if self.cg_type == "FR":
                beta_num = self._scalar_product(q, q)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                beta = beta_num / beta_denom
            elif self.cg_type == "PR":
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                beta = beta_num / beta_denom
            elif self.cg_type == "HS":
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = -self._scalar_product(out, self.difference)
                beta = beta_num / beta_denom
            elif self.cg_type == "DY":
                beta_num = self._scalar_product(q, q)
                beta_denom = -self._scalar_product(out, self.difference)
                beta = beta_num / beta_denom
            elif self.cg_type == "HZ":
                dy = -self._scalar_product(out, self.difference)
                y2 = self._scalar_product(self.difference, self.difference)

                self.difference[0].vector().vec().axpby(
                    -2 * y2 / dy, -1.0, out[0].vector().vec()
                )
                self.difference[0].vector().apply("")
                beta = -self._scalar_product(self.difference, q) / dy
            else:
                beta = 0.0
        else:
            beta = 0.0

        out[0].vector().vec().aypx(beta, q[0].vector().vec())
        out[0].vector().apply("")

    def _compute_eps(self) -> float:
        """Computes and returns the termination parameter epsilon."""
        self.diff[0].vector().vec().aypx(
            0.0, self.p_current[0].vector().vec() - self.z_star[0].vector().vec()
        )
        self.diff[0].vector().apply("")
        eps = float(
            np.sqrt(self._scalar_product(self.diff, self.diff)) / self.norm_z_star
        )

        return eps

    def inject_pre_callback(self, function: Optional[Callable]) -> None:
        """Changes the a-priori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self.coarse_model._pre_callback = function  # pylint: disable=protected-access
        # pylint: disable=protected-access
        self.parameter_extraction._pre_callback = function

    def inject_post_callback(self, function: Optional[Callable]) -> None:
        """Changes the a-posteriori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self.coarse_model._post_callback = function  # pylint: disable=protected-access
        # pylint: disable=protected-access
        self.parameter_extraction._post_callback = function

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
