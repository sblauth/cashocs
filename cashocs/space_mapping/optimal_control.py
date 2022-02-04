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
from typing import Union, List, Dict, Optional

from typing_extensions import Literal

import numpy as np
import fenics
import ufl
import collections
import configparser

from cashocs._optimization.optimal_control import optimal_control_problem as ocp
from cashocs import _exceptions
from cashocs import utils


class ParentFineModel(abc.ABC):
    """Base class for the fine model in space mapping.

    Attributes:
        controls: The control variables of the fine model.
        cost_functional_value: The current cost functional value of the fine model.
    """

    def __init__(self) -> None:
        """Initializes self."""

        self.controls = None
        self.cost_functional_value = None

    @abc.abstractmethod
    def solve_and_evaluate(self) -> None:
        """Solves and evaluates the fine model.

        This needs to be overwritten with a custom implementation.
        """

        pass


class CoarseModel:
    """Coarse Model for space mapping."""

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
        self.scalar_tracking_forms = scalar_tracking_forms
        self.min_max_terms = min_max_terms
        self.desired_weights = desired_weights

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
            scalar_tracking_forms=self.scalar_tracking_forms,
            min_max_terms=self.min_max_terms,
            desired_weights=self.desired_weights,
        )

    def optimize(self) -> None:

        self.optimal_control_problem.solve()


class ParameterExtraction:
    def __init__(
        self,
        coarse_model: CoarseModel,
        cost_functional_form: Union[ufl.Form, List[ufl.Form]],
        states: Union[fenics.Function, List[fenics.Function]],
        controls: Union[fenics.Function, List[fenics.Function]],
        config: Optional[configparser.ConfigParser] = None,
        scalar_tracking_forms: Optional[Union[Dict, List[Dict]]] = None,
        desired_weights: Optional[List[float]] = None,
    ) -> None:

        self.coarse_model = coarse_model
        self.cost_functional_form = cost_functional_form

        self.states = utils.enlist(states)
        self.controls = utils.enlist(controls)

        self.config = config
        self.scalar_tracking_forms = scalar_tracking_forms
        self.desired_weights = desired_weights

        self.adjoints = [
            fenics.Function(V)
            for V in coarse_model.optimal_control_problem.form_handler.adjoint_spaces
        ]

        dict_states = {
            coarse_model.optimal_control_problem.states[i]: self.states[i]
            for i in range(len(self.states))
        }
        dict_adjoints = {
            coarse_model.optimal_control_problem.adjoints[i]: self.adjoints[i]
            for i in range(len(self.adjoints))
        }
        dict_controls = {
            coarse_model.optimal_control_problem.controls[i]: self.controls[i]
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
            coarse_model.optimal_control_problem.control_constraints
        )
        self.initial_guess = coarse_model.optimal_control_problem.initial_guess
        self.ksp_options = coarse_model.optimal_control_problem.ksp_options
        self.adjoint_ksp_options = (
            coarse_model.optimal_control_problem.adjoint_ksp_options
        )

        self.optimal_control_problem = None

    def _solve(self, initial_guesses: Optional[List[fenics.Function]] = None) -> None:

        if initial_guesses is None:
            for i in range(len(self.controls)):
                self.controls[i].vector()[:] = 0.0
        else:
            for i in range(len(self.controls)):
                self.controls[i].vector()[:] = initial_guesses[i].vector()[:]

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
            scalar_tracking_forms=self.scalar_tracking_forms,
        )

        self.optimal_control_problem.solve()


class SpaceMapping:
    def __init__(
        self,
        fine_model: ParentFineModel,
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
    ) -> None:

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

        self.eps = 1.0
        self.converged = False
        self.iteration = 0

        self.z_star = self.coarse_model.optimal_control_problem.controls
        self.norm_z_star = 1.0
        self.control_dim = (
            self.coarse_model.optimal_control_problem.form_handler.control_dim
        )

        self.x = utils.enlist(self.fine_model.controls)

        control_spaces_fine = [xx.function_space() for xx in self.x]
        control_spaces_coarse = (
            self.coarse_model.optimal_control_problem.form_handler.control_spaces
        )
        self.ips_to_coarse = [
            utils.Interpolator(control_spaces_fine[i], control_spaces_coarse[i])
            for i in range(len(self.z_star))
        ]
        self.ips_to_fine = [
            utils.Interpolator(control_spaces_coarse[i], control_spaces_fine[i])
            for i in range(len(self.z_star))
        ]

        self.p_current = self.parameter_extraction.controls
        self.p_prev = [fenics.Function(V) for V in control_spaces_coarse]
        self.h = [fenics.Function(V) for V in control_spaces_coarse]
        self.v = [fenics.Function(V) for V in control_spaces_coarse]
        self.u = [fenics.Function(V) for V in control_spaces_coarse]

        self.x_save = [fenics.Function(V) for V in control_spaces_fine]

        self.diff = [fenics.Function(V) for V in control_spaces_coarse]
        self.temp = [fenics.Function(V) for V in control_spaces_coarse]
        self.dir_prev = [fenics.Function(V) for V in control_spaces_coarse]
        self.difference = [fenics.Function(V) for V in control_spaces_coarse]

        self.history_s = collections.deque()
        self.history_y = collections.deque()
        self.history_rho = collections.deque()
        self.history_alpha = collections.deque()

    def solve(self) -> None:
        # Compute initial guess for the space mapping
        self.coarse_model.optimize()
        for i in range(self.control_dim):
            self.x[i].vector()[:] = (
                self.scaling_factor
                * self.ips_to_fine[i].interpolate(self.z_star[i]).vector()[:]
            )
        self.norm_z_star = np.sqrt(self._scalar_product(self.z_star, self.z_star))

        self.fine_model.solve_and_evaluate()
        self.parameter_extraction._solve(
            initial_guesses=[
                self.ips_to_coarse[i].interpolate(self.x[i])
                for i in range(self.control_dim)
            ]
        )
        self.eps = self._compute_eps()

        if self.verbose:
            print(
                f"Space Mapping - Iteration {self.iteration:3d}:    "
                f"Cost functional value = "
                f"{self.fine_model.cost_functional_value:.3e}    "
                f"eps = {self.eps:.3e}\n"
            )

        while not self.converged:
            for i in range(self.control_dim):
                self.dir_prev[i].vector()[:] = -(
                    self.p_prev[i].vector()[:] - self.z_star[i].vector()[:]
                )
                self.temp[i].vector()[:] = -(
                    self.p_current[i].vector()[:] - self.z_star[i].vector()[:]
                )

            self._compute_search_direction(self.temp, self.h)

            stepsize = 1.0
            for i in range(self.control_dim):
                self.p_prev[i].vector()[:] = self.p_current[i].vector()[:]
            if not self.use_backtracking_line_search:
                for i in range(self.control_dim):
                    self.x[i].vector()[:] += (
                        self.scaling_factor
                        * self.ips_to_fine[i].interpolate(self.h[i]).vector()[:]
                    )

                self.fine_model.solve_and_evaluate()
                self.parameter_extraction._solve(
                    initial_guesses=[
                        self.ips_to_coarse[i].interpolate(self.x[i])
                        for i in range(self.control_dim)
                    ]
                )
                self.eps = self._compute_eps()

            else:
                for i in range(self.control_dim):
                    self.x_save[i].vector()[:] = self.x[i].vector()[:]

                while True:
                    for i in range(self.control_dim):
                        self.x[i].vector()[:] = self.x_save[i].vector()[:]
                        self.x[i].vector()[:] += (
                            self.scaling_factor
                            * stepsize
                            * self.ips_to_fine[i].interpolate(self.h[i]).vector()[:]
                        )
                    self.fine_model.solve_and_evaluate()
                    self.parameter_extraction._solve(
                        initial_guesses=[
                            self.ips_to_coarse[i].interpolate(self.x[i])
                            for i in range(self.control_dim)
                        ]
                    )
                    eps_new = self._compute_eps()

                    if eps_new <= self.eps:
                        self.eps = eps_new
                        break
                    else:
                        stepsize /= 2

                    if stepsize <= 1e-4:
                        raise _exceptions.NotConvergedError(
                            "Space Mapping Backtracking Line Search",
                            "The line search did not converge.",
                        )

            self.iteration += 1
            if self.verbose:
                print(
                    f"Space Mapping - Iteration {self.iteration:3d}:    "
                    f"Cost functional value = "
                    f"{self.fine_model.cost_functional_value:.3e}    "
                    f"eps = {self.eps:.3e}"
                    f"    step size = {stepsize:.3e}"
                )

            if self.eps <= self.tol:
                self.converged = True
                break
            if self.iteration >= self.max_iter:
                break

            if self.method == "broyden":
                for i in range(self.control_dim):
                    self.temp[i].vector()[:] = (
                        self.p_current[i].vector()[:] - self.p_prev[i].vector()[:]
                    )
                self._compute_broyden_application(self.temp, self.v)

                if self.memory_size > 0:
                    if self.broyden_type == "good":
                        divisor = self._scalar_product(self.h, self.v)
                        for i in range(self.control_dim):
                            self.u[i].vector()[:] = (
                                self.h[i].vector()[:] - self.v[i].vector()[:]
                            ) / divisor

                        self.history_s.append([xx.copy(True) for xx in self.u])
                        self.history_y.append([xx.copy(True) for xx in self.h])

                    elif self.broyden_type == "bad":
                        divisor = self._scalar_product(self.temp, self.temp)
                        for i in range(self.control_dim):
                            self.u[i].vector()[:] = (
                                self.h[i].vector()[:] - self.v[i].vector()[:]
                            ) / divisor

                            self.history_s.append([xx.copy(True) for xx in self.u])
                            self.history_y.append([xx.copy(True) for xx in self.temp])

                    if len(self.history_s) > self.memory_size:
                        self.history_s.popleft()
                        self.history_y.popleft()

            elif self.method == "bfgs":
                if self.memory_size > 0:
                    for i in range(self.control_dim):
                        self.temp[i].vector()[:] = (
                            self.p_current[i].vector()[:] - self.p_prev[i].vector()[:]
                        )

                    self.history_y.appendleft([xx.copy(True) for xx in self.temp])
                    self.history_s.appendleft([xx.copy(True) for xx in self.h])
                    curvature_condition = self._scalar_product(self.temp, self.h)

                    if curvature_condition <= 0:
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

        if self.converged:
            output = (
                f"\nStatistics --- "
                f"Space mapping iterations: {self.iteration:4d} --- "
                f"Final objective value: {self.fine_model.cost_functional_value:.3e}\n"
            )
            if self.verbose:
                print(output)

    def _scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:

        return self.coarse_model.optimal_control_problem.form_handler.scalar_product(
            a, b
        )

    def _compute_search_direction(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:

        if self.method == "steepest_descent":
            return self._compute_steepest_descent_application(q, out)
        elif self.method == "broyden":
            return self._compute_broyden_application(q, out)
        elif self.method == "bfgs":
            return self._compute_bfgs_application(q, out)
        elif self.method == "ncg":
            return self._compute_ncg_direction(q, out)

    def _compute_steepest_descent_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        for i in range(self.control_dim):
            out[i].vector()[:] = q[i].vector()[:]

    def _compute_broyden_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        for j in range(self.control_dim):
            out[j].vector()[:] = q[j].vector()[:]

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

            for j in range(self.control_dim):
                out[j].vector()[:] += alpha * self.history_s[i][j].vector()[:]

    def _compute_bfgs_application(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        for j in range(self.control_dim):
            out[j].vector()[:] = q[j].vector()[:]

        if len(self.history_s) > 0:
            self.history_alpha.clear()

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self._scalar_product(
                    self.history_s[i], out
                )
                self.history_alpha.append(alpha)
                for j in range(self.control_dim):
                    out[j].vector()[:] -= alpha * self.history_y[i][j].vector()[:]

            bfgs_factor = self._scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self._scalar_product(self.history_y[0], self.history_y[0])
            for j in range(self.control_dim):
                out[j].vector()[:] *= bfgs_factor

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self._scalar_product(
                    self.history_y[-1 - i], out
                )
                for j in range(self.control_dim):
                    out[j].vector()[:] += self.history_s[-1 - i][j].vector()[:] * (
                        self.history_alpha[-1 - i] - beta
                    )

    def _compute_ncg_direction(
        self, q: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        if self.iteration > 0:
            if self.cg_type == "FR":
                beta_num = self._scalar_product(q, q)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                self.beta = beta_num / beta_denom

            elif self.cg_type == "PR":
                for i in range(self.control_dim):
                    self.difference[i].vector()[:] = (
                        q[i].vector()[:] - self.dir_prev[i].vector()[:]
                    )
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                self.beta = beta_num / beta_denom

            elif self.cg_type == "HS":
                for i in range(self.control_dim):
                    self.difference[i].vector()[:] = (
                        q[i].vector()[:] - self.dir_prev[i].vector()[:]
                    )
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = -self._scalar_product(out, self.difference)
                self.beta = beta_num / beta_denom

            elif self.cg_type == "DY":
                for i in range(self.control_dim):
                    self.difference[i].vector()[:] = (
                        q[i].vector()[:] - self.dir_prev[i].vector()[:]
                    )
                beta_num = self._scalar_product(q, q)
                beta_denom = -self._scalar_product(out, self.difference)
                self.beta = beta_num / beta_denom

            elif self.cg_type == "HZ":
                for i in range(self.control_dim):
                    self.difference[i].vector()[:] = (
                        q[i].vector()[:] - self.dir_prev[i].vector()[:]
                    )
                dy = -self._scalar_product(out, self.difference)
                y2 = self._scalar_product(self.difference, self.difference)

                for i in range(self.control_dim):
                    self.difference[i].vector()[:] = (
                        -self.difference[i].vector()[:]
                        - 2 * y2 / dy * out[i].vector()[:]
                    )
                self.beta = -self._scalar_product(self.difference, q) / dy

        else:
            self.beta = 0.0

        for i in range(self.control_dim):
            out[i].vector()[:] = q[i].vector()[:] + self.beta * out[i].vector()[:]

    def _compute_eps(self) -> float:

        for i in range(self.control_dim):
            self.diff[i].vector()[:] = (
                self.p_current[i].vector()[:] - self.z_star[i].vector()[:]
            )
            eps = np.sqrt(self._scalar_product(self.diff, self.diff)) / self.norm_z_star

            return eps
