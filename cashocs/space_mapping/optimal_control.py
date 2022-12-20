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

"""Space mapping for optimal control problems."""

from __future__ import annotations

import abc
import collections
import json
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np
from typing_extensions import Literal
import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs._optimization.optimal_control import optimal_control_problem as ocp

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io


class FineModel(abc.ABC):
    """Base class for the fine model in space mapping.

    Attributes:
        controls: The control variables of the fine model.
        cost_functional_value: The current cost functional value of the fine model.

    """

    controls: List[fenics.Function]
    cost_functional_value: float

    def __init__(self) -> None:
        """Initializes self."""
        pass

    @abc.abstractmethod
    def solve_and_evaluate(self) -> None:
        """Solves and evaluates the fine model.

        This needs to be overwritten with a custom implementation.

        """
        pass


class CoarseModel:
    """Coarse Model for space mapping optimal control."""

    def __init__(
        self,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
        cost_functional_form: Union[
            List[_typing.CostFunctional], _typing.CostFunctional
        ],
        states: Union[List[fenics.Function], fenics.Function],
        controls: Union[List[fenics.Function], fenics.Function],
        adjoints: Union[List[fenics.Function], fenics.Function],
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
        desired_weights: Optional[List[float]] = None,
    ) -> None:
        """Initializes self.

        Args:
            state_forms: The list of weak forms for the coare state problem
            bcs_list: The list of boundary conditions for the coarse problem
            cost_functional_form: The cost functional for the coarse problem
            states: The state variables for the coarse problem
            controls: The control variables for the coarse problem
            adjoints: The adjoint variables for the coarse problem
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            riesz_scalar_products: The scalar products for the coarse problem
            control_constraints: The box constraints for the problem
            initial_guess: The initial guess for solving a nonlinear state equation
            ksp_options: The list of PETSc options for the state equations
            adjoint_ksp_options: The list of PETSc options for the adjoint equations
            desired_weights: The desired weights for the cost functional

        """
        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.cost_functional_form = cost_functional_form
        self.states = states
        self.controls = controls
        self.adjoints = adjoints
        self.config = config
        self.riesz_scalar_products = riesz_scalar_products
        self.control_constraints = control_constraints
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.desired_weights = desired_weights

        self._pre_callback: Optional[Callable] = None
        self._post_callback: Optional[Callable] = None

        self.optimal_control_problem = ocp.OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.controls,
            self.adjoints,
            config=self.config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
        )

    def optimize(self) -> None:
        """Solves the coarse model optimization problem."""
        self.optimal_control_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )
        self.optimal_control_problem.solve()


class ParameterExtraction:
    """Parameter extraction for optimal control problems."""

    def __init__(
        self,
        coarse_model: CoarseModel,
        cost_functional_form: Union[
            List[_typing.CostFunctional], _typing.CostFunctional
        ],
        states: Union[List[fenics.Function], fenics.Function],
        controls: Union[List[fenics.Function], fenics.Function],
        config: Optional[io.Config] = None,
        desired_weights: Optional[List[float]] = None,
        mode: str = "initial",
    ) -> None:
        """Initializes self.

        Args:
            coarse_model: The coarse model optimization problem
            cost_functional_form: The cost functional for the parameter extraction
            states: The state variables for the parameter extraction
            controls: The control variables for the parameter extraction
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
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
        self.cost_functional_form = cost_functional_form

        self.states = _utils.enlist(states)
        self.controls: List[fenics.Function] = _utils.enlist(controls)

        self.config = config
        self.mode = mode
        self.desired_weights = desired_weights

        self._pre_callback: Optional[Callable] = None
        self._post_callback: Optional[Callable] = None

        self.adjoints = _utils.create_function_list(
            coarse_model.optimal_control_problem.db.function_db.adjoint_spaces
        )

        dict_states = {
            coarse_model.optimal_control_problem.states[i]: self.states[i]
            for i in range(len(self.states))
        }
        dict_adjoints = {
            coarse_model.optimal_control_problem.adjoints[i]: self.adjoints[i]
            for i in range(len(self.adjoints))
        }
        dict_controls = {
            coarse_model.optimal_control_problem.db.function_db.controls[
                i
            ]: self.controls[i]
            for i in range(len(self.controls))
        }
        mapping_dict = {}
        mapping_dict.update(dict_states)
        mapping_dict.update(dict_adjoints)
        mapping_dict.update(dict_controls)
        self.state_forms = [
            ufl.replace(form, mapping_dict)
            for form in coarse_model.optimal_control_problem.state_forms
        ]
        self.bcs_list = coarse_model.optimal_control_problem.bcs_list
        self.riesz_scalar_products = (
            coarse_model.optimal_control_problem.riesz_scalar_products
        )
        self.control_constraints = (
            coarse_model.optimal_control_problem.box_constraints.control_constraints
        )
        self.initial_guess = coarse_model.optimal_control_problem.initial_guess
        self.ksp_options = coarse_model.optimal_control_problem.ksp_options
        self.adjoint_ksp_options = (
            coarse_model.optimal_control_problem.adjoint_ksp_options
        )
        self.optimal_control_problem: Optional[ocp.OptimalControlProblem] = None

    def _solve(self, initial_guesses: Optional[List[fenics.Function]] = None) -> None:
        """Solves the parameter extraction problem.

        Args:
            initial_guesses: The list of initial guesses for solving the problem.

        """
        if self.mode == "initial":
            for i in range(len(self.controls)):
                self.controls[i].vector().vec().set(0.0)
                self.controls[i].vector().apply("")
        elif self.mode == "coarse_optimum" and initial_guesses is not None:
            for i in range(len(self.controls)):
                self.controls[i].vector().vec().aypx(
                    0.0, initial_guesses[i].vector().vec()
                )
                self.controls[i].vector().apply("")
        else:
            raise _exceptions.InputError(
                "ParameterExtraction._solve", "initial_guesses", ""
            )

        self.optimal_control_problem = ocp.OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.controls,
            self.adjoints,
            config=self.config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
        )

        self.optimal_control_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )
        self.optimal_control_problem.solve()


class SpaceMappingProblem:
    """Space mapping method for optimal control problems."""

    def __init__(
        self,
        fine_model: FineModel,
        coarse_model: CoarseModel,
        parameter_extraction: ParameterExtraction,
        method: Literal[
            "broyden", "bfgs", "lbfgs", "sd", "steepest_descent", "ncg"
        ] = "broyden",
        max_iter: int = 25,
        tol: float = 1e-2,
        use_backtracking_line_search: bool = False,
        broyden_type: Literal["good", "bad"] = "good",
        cg_type: Literal["FR", "PR", "HS", "DY", "HZ"] = "FR",
        memory_size: int = 10,
        scaling_factor: float = 1.0,
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
            scaling_factor: A factor, which can be used to appropriately scale the
                control variables between fine and coarse model.
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
        self.scaling_factor = scaling_factor
        self.verbose = verbose
        self.save_history = save_history

        self.eps = 1.0
        self.converged = False
        self.iteration = 0
        self.stepsize = 1.0

        self.z_star = self.coarse_model.optimal_control_problem.db.function_db.controls
        self.norm_z_star = 1.0

        self.x: List[fenics.Function] = _utils.enlist(self.fine_model.controls)

        control_spaces_fine = [xx.function_space() for xx in self.x]
        control_spaces_coarse = (
            self.coarse_model.optimal_control_problem.db.function_db.control_spaces
        )
        self.ips_to_coarse = [
            _utils.Interpolator(control_spaces_fine[i], control_spaces_coarse[i])
            for i in range(len(self.z_star))
        ]
        self.ips_to_fine = [
            _utils.Interpolator(control_spaces_coarse[i], control_spaces_fine[i])
            for i in range(len(self.z_star))
        ]

        self.p_current = self.parameter_extraction.controls
        self.p_prev = _utils.create_function_list(control_spaces_coarse)
        self.h = _utils.create_function_list(control_spaces_coarse)
        self.v = _utils.create_function_list(control_spaces_coarse)
        self.u = _utils.create_function_list(control_spaces_coarse)

        self.x_save = _utils.create_function_list(control_spaces_fine)

        self.diff = _utils.create_function_list(control_spaces_coarse)
        self.temp = _utils.create_function_list(control_spaces_coarse)
        self.dir_prev = _utils.create_function_list(control_spaces_coarse)
        self.difference = _utils.create_function_list(control_spaces_coarse)

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

    def _compute_intial_guess(self) -> None:
        """Compute initial guess for the space mapping by solving the coarse problem."""
        self.coarse_model.optimize()
        for i in range(len(self.x)):
            self.x[i].vector().vec().aypx(
                0.0,
                self.scaling_factor
                * self.ips_to_fine[i].interpolate(self.z_star[i]).vector().vec(),
            )
            self.x[i].vector().apply("")
        self.norm_z_star = np.sqrt(self._scalar_product(self.z_star, self.z_star))

    def test_for_nonconvergence(self) -> None:
        """Tests, whether maximum number of iterations are exceeded."""
        if self.iteration >= self.max_iter:
            raise _exceptions.NotConvergedError(
                "Space Mapping",
                "Maximum number of iterations exceeded.",
            )

    def solve(self) -> None:
        """Solves the space mapping problem."""
        self._compute_intial_guess()

        self.fine_model.solve_and_evaluate()
        self.parameter_extraction._solve()  # pylint: disable=protected-access
        self.eps = self._compute_eps()

        self.update_history()
        if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            print(
                f"Space Mapping - Iteration {self.iteration:3d}:    "
                f"Cost functional value = "
                f"{self.fine_model.cost_functional_value:.3e}    "
                f"eps = {self.eps:.3e}\n",
                flush=True,
            )
        fenics.MPI.barrier(fenics.MPI.comm_world)

        while not self.converged:
            for i in range(len(self.dir_prev)):
                self.dir_prev[i].vector().vec().aypx(
                    0.0,
                    -(self.p_prev[i].vector().vec() - self.z_star[i].vector().vec()),
                )
                self.dir_prev[i].vector().apply("")
                self.temp[i].vector().vec().aypx(
                    0.0,
                    -(self.p_current[i].vector().vec() - self.z_star[i].vector().vec()),
                )
                self.temp[i].vector().apply("")

            self._compute_search_direction(self.temp, self.h)

            for i in range(len(self.p_prev)):
                self.p_prev[i].vector().vec().aypx(
                    0.0, self.p_current[i].vector().vec()
                )
                self.p_prev[i].vector().apply("")

            self._update_iterates()

            self.iteration += 1
            self.update_history()
            if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                print(
                    f"Space Mapping - Iteration {self.iteration:3d}:    "
                    f"Cost functional value = "
                    f"{self.fine_model.cost_functional_value:.3e}    "
                    f"eps = {self.eps:.3e}"
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
                    json.dump(self.space_mapping_history, file)
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
            for i in range(len(self.temp)):
                self.temp[i].vector().vec().aypx(
                    0.0,
                    self.p_current[i].vector().vec() - self.p_prev[i].vector().vec(),
                )
                self.temp[i].vector().apply("")
            self._compute_broyden_application(self.temp, self.v)

            if self.memory_size > 0:
                if self.broyden_type == "good":
                    divisor = self._scalar_product(self.h, self.v)
                    for i in range(len(self.u)):
                        self.u[i].vector().vec().aypx(
                            0.0,
                            (self.h[i].vector().vec() - self.v[i].vector().vec())
                            / divisor,
                        )
                        self.u[i].vector().apply("")

                    self.history_s.append([xx.copy(True) for xx in self.u])
                    self.history_y.append([xx.copy(True) for xx in self.h])

                elif self.broyden_type == "bad":
                    divisor = self._scalar_product(self.temp, self.temp)
                    for i in range(len(self.u)):
                        self.u[i].vector().vec().aypx(
                            0.0,
                            (self.h[i].vector().vec() - self.v[i].vector().vec())
                            / divisor,
                        )
                        self.u[i].vector().apply("")

                        self.history_s.append([xx.copy(True) for xx in self.u])
                        self.history_y.append([xx.copy(True) for xx in self.temp])

                if len(self.history_s) > self.memory_size:
                    self.history_s.popleft()
                    self.history_y.popleft()

    def _update_bfgs_approximation(self) -> None:
        """Updates the approximation of the mapping function with the BFGS method."""
        if self.method == "bfgs":
            if self.memory_size > 0:
                for i in range(len(self.temp)):
                    self.temp[i].vector().vec().aypx(
                        0.0,
                        self.p_current[i].vector().vec()
                        - self.p_prev[i].vector().vec(),
                    )
                    self.temp[i].vector().apply("")

                self.history_y.appendleft([xx.copy(True) for xx in self.temp])
                self.history_s.appendleft([xx.copy(True) for xx in self.h])
                curvature_condition = self._scalar_product(self.temp, self.h)

                if curvature_condition <= 0:
                    self.history_s.clear()
                    self.history_y.clear()
                    self.history_rho.clear()

                else:
                    rho = 1.0 / curvature_condition
                    self.history_rho.appendleft(rho)

                if len(self.history_s) > self.memory_size:
                    self.history_s.pop()
                    self.history_y.pop()
                    self.history_rho.pop()

    def _update_iterates(self) -> None:
        """Updates the iterates either directly or via a line search."""
        self.stepsize = 1.0
        if not self.use_backtracking_line_search:
            for i in range(len(self.x)):
                self.x[i].vector().vec().axpy(
                    self.scaling_factor,
                    self.ips_to_fine[i].interpolate(self.h[i]).vector().vec(),
                )
                self.x[i].vector().apply("")

            self.fine_model.solve_and_evaluate()
            self.parameter_extraction._solve(  # pylint: disable=protected-access
                initial_guesses=[
                    ips.interpolate(self.x[i])
                    for i, ips in enumerate(self.ips_to_coarse)
                ]
            )
            self.eps = self._compute_eps()

        else:
            for i in range(len(self.x_save)):
                self.x_save[i].vector().vec().aypx(0.0, self.x[i].vector().vec())
                self.x_save[i].vector().apply("")

            while True:
                for i in range(len(self.x)):
                    self.x[i].vector().vec().aypx(0.0, self.x_save[i].vector().vec())
                    self.x[i].vector().apply("")
                    self.x[i].vector().vec().axpy(
                        self.scaling_factor * self.stepsize,
                        self.ips_to_fine[i].interpolate(self.h[i]).vector().vec(),
                    )
                    self.x[i].vector().apply("")
                self.fine_model.solve_and_evaluate()
                self.parameter_extraction._solve(  # pylint: disable=protected-access
                    initial_guesses=[
                        ips.interpolate(self.x[i])
                        for i, ips in enumerate(self.ips_to_coarse)
                    ]
                )
                eps_new = self._compute_eps()

                if eps_new <= self.eps:
                    self.eps = eps_new
                    break
                else:
                    self.stepsize /= 2

                if self.stepsize <= 1e-4:
                    raise _exceptions.NotConvergedError(
                        "Space Mapping Backtracking Line Search",
                        "The line search did not converge.",
                    )

    def _scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between ``a`` and ``b``.

        Args:
            a: The first input for the scalar product
            b: The second input for the scalar product

        Returns:
            The scalar product of ``a`` and ``b``

        """
        return self.coarse_model.optimal_control_problem.form_handler.scalar_product(
            a, b
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
                "cashocs.space_mapping.optimal_control.SpaceMapping",
                "method",
                "The method is not supported.",
            )

    def _compute_steepest_descent_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
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
        for j in range(len(out)):
            out[j].vector().vec().aypx(0.0, q[j].vector().vec())
            out[j].vector().apply("")

        for i in range(len(self.history_s)):
            if self.broyden_type == "good":
                alpha = self._scalar_product(self.history_y[i], out)
            elif self.broyden_type == "bad":
                alpha = self._scalar_product(self.history_y[i], q)
            else:
                raise _exceptions.InputError(
                    "cashocs.space_mapping.optimal_control.SpaceMapping",
                    "broyden_type",
                    "broyden_type has to be either 'good' or 'bad'.",
                )

            for j in range(len(out)):
                out[j].vector().vec().axpy(alpha, self.history_s[i][j].vector().vec())
                out[j].vector().apply("")

    def _compute_bfgs_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for the LBFGS method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        for j in range(len(out)):
            out[j].vector().vec().aypx(0.0, q[j].vector().vec())
            out[j].vector().apply("")

        if len(self.history_s) > 0:
            self.history_alpha.clear()

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self._scalar_product(
                    self.history_s[i], out
                )
                self.history_alpha.append(alpha)
                for j in range(len(out)):
                    out[j].vector().vec().axpy(
                        -alpha, self.history_y[i][j].vector().vec()
                    )
                    out[j].vector().apply("")

            bfgs_factor = self._scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self._scalar_product(self.history_y[0], self.history_y[0])
            for j in range(len(out)):
                out[j].vector().vec().scale(bfgs_factor)
                out[j].vector().apply("")

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self._scalar_product(
                    self.history_y[-1 - i], out
                )
                for j in range(len(out)):
                    out[j].vector().vec().axpy(
                        self.history_alpha[-1 - i] - beta,
                        self.history_s[-1 - i][j].vector().vec(),
                    )
                    out[j].vector().apply("")

    def _compute_ncg_direction(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the search direction for the NCG methods.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.iteration > 0:
            for i in range(len(self.difference)):
                self.difference[i].vector().vec().aypx(
                    0.0, q[i].vector().vec() - self.dir_prev[i].vector().vec()
                )
                self.difference[i].vector().apply("")

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

                for i in range(len(self.difference)):
                    self.difference[i].vector().vec().aypx(
                        0.0,
                        -self.difference[i].vector().vec()
                        - 2 * y2 / dy * out[i].vector().vec(),
                    )
                    self.difference[i].vector().apply("")
                beta = -self._scalar_product(self.difference, q) / dy
            else:
                beta = 0.0
        else:
            beta = 0.0

        for i in range(len(out)):
            out[i].vector().vec().aypx(beta, q[i].vector().vec())
            out[i].vector().apply("")

    def _compute_eps(self) -> float:
        """Computes and returns the termination parameter epsilon."""
        for i in range(len(self.diff)):
            self.diff[i].vector().vec().aypx(
                0.0, self.p_current[i].vector().vec() - self.z_star[i].vector().vec()
            )
            self.diff[i].vector().apply("")
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
