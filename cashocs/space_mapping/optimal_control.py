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

"""Space mapping for optimal control problems

"""

import numpy as np
import fenics
from ufl import replace
from _collections import deque

from .._optimal_control.optimal_control_problem import OptimalControlProblem
from .._exceptions import InputError, NotConvergedError
from ..utils import (
    _check_and_enlist_functions,
    _check_and_enlist_ufl_forms,
    Interpolator,
)
from .._loggers import debug


class ParentFineModel:
    def __init__(self):
        self.controls = None
        self.cost_functional_value = None

    def solve_and_evaluate(self):
        pass


class CoarseModel:
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        controls,
        adjoints,
        config=None,
        riesz_scalar_products=None,
        control_constraints=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        desired_weights=None,
        scalar_tracking_forms=None,
    ):

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
        self.scalar_tracking_forms = scalar_tracking_forms

        self.optimal_control_problem = OptimalControlProblem(
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

    def optimize(self):

        self.optimal_control_problem.solve()


class ParameterExtraction:
    def __init__(
        self,
        coarse_model,
        cost_functional_form,
        states,
        controls,
        config=None,
        scalar_tracking_forms=None,
        desired_weights=None,
    ):
        """

        Parameters
        ----------
        coarse_model : CoarseModel
        cost_functional_form
        states
        controls
        config
        scalar_tracking_forms
        desired_weights

        Returns
        -------

        """

        self.coarse_model = coarse_model
        self.cost_functional_form = cost_functional_form

        ### states
        try:
            self.states = _check_and_enlist_functions(states)
        except InputError:
            raise InputError(
                "cashocs.space_mapping.optimal_control.ParameterExtraction",
                "states",
                "Type of states is wrong.",
            )

        ### controls
        try:
            self.controls = _check_and_enlist_functions(controls)
        except InputError:
            raise InputError(
                "cashocs.space_mapping.optimal_control.ParameterExtraction",
                "controls",
                "Type of controls is wrong.",
            )

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
            replace(form, mapping_dict)
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

    def _solve(self, initial_guesses=None):

        if initial_guesses is None:
            for i in range(len(self.controls)):
                self.controls[i].vector()[:] = 0.0
        else:
            for i in range(len(self.controls)):
                self.controls[i].vector()[:] = initial_guesses[i].vector()[:]

        self.optimal_control_problem = OptimalControlProblem(
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
        fine_model,
        coarse_model,
        parameter_extraction,
        method="broyden",
        max_iter=25,
        tol=1e-2,
        use_backtracking_line_search=False,
        broyden_type="good",
        cg_type="FR",
        memory_size=10,
        scaling_factor=1.0,
        verbose=True,
    ):
        """

        Parameters
        ----------
        fine_model : ParentFineModel
        coarse_model : CoarseModel
        parameter_extraction : ParameterExtraction
        method
        max_iter
        tol
        use_backtracking_line_search
        broyden_type
        cg_type
        memory_size
        scaling_factor
        verbose
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

        self.eps = 1.0
        self.converged = False
        self.iteration = 0

        self.z_star = self.coarse_model.optimal_control_problem.controls
        self.control_dim = (
            self.coarse_model.optimal_control_problem.form_handler.control_dim
        )
        try:
            self.x = _check_and_enlist_functions(self.fine_model.controls)
        except InputError:
            raise InputError(
                "cashocs.space_mapping.optimal_control.ParentFineModel",
                "self.controls",
                "The parameter self.controls has to be defined either as a single "
                + "fenics Function or a list of fenics Functions.",
            )

        control_spaces_fine = [xx.function_space() for xx in self.x]
        control_spaces_coarse = (
            self.coarse_model.optimal_control_problem.form_handler.control_spaces
        )
        self.ips_to_coarse = [
            Interpolator(control_spaces_fine[i], control_spaces_coarse[i])
            for i in range(len(self.z_star))
        ]
        self.ips_to_fine = [
            Interpolator(control_spaces_coarse[i], control_spaces_fine[i])
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

        self.history_s = deque()
        self.history_y = deque()
        self.history_rho = deque()
        self.history_alpha = deque()

    def solve(self):
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
                f"Space Mapping - Iteration {self.iteration:3d}:    Cost functional value = "
                f"{self.fine_model.cost_functional_value:.3e}    eps = {self.eps:.3e}\n"
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

            # if self._scalar_product(self.h, self.temp) <= 0.0:
            #     debug(
            #         "The computed search direction for space mapping did not yield a descent direction"
            #     )
            #     for i in range(self.control_dim):
            #         self.h[i].vector()[:] = self.temp[i].vector()[:]

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
                        raise NotConvergedError(
                            "Space Mapping Backtracking Line Search",
                            "The line search did not converge.",
                        )

            self.iteration += 1
            if self.verbose:
                print(
                    f"Space Mapping - Iteration {self.iteration:3d}:    Cost functional value = "
                    f"{self.fine_model.cost_functional_value:.3e}    eps = {self.eps:.3e}"
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
                f"\nStatistics --- Space mapping iterations: {self.iteration:4d}"
                + f" --- Final objective value: {self.fine_model.cost_functional_value:.3e}\n"
            )
            if self.verbose:
                print(output)

    def _scalar_product(self, a, b):
        return self.coarse_model.optimal_control_problem.form_handler.scalar_product(
            a, b
        )

    def _compute_search_direction(self, q, out):
        if self.method == "steepest_descent":
            return self._compute_steepest_descent_application(q, out)
        elif self.method == "broyden":
            return self._compute_broyden_application(q, out)
        elif self.method == "bfgs":
            return self._compute_bfgs_application(q, out)
        elif self.method == "ncg":
            return self._compute_ncg_direction(q, out)

    def _compute_steepest_descent_application(self, q, out):
        for i in range(self.control_dim):
            out[i].vector()[:] = q[i].vector()[:]

    def _compute_broyden_application(self, q, out):
        for j in range(self.control_dim):
            out[j].vector()[:] = q[j].vector()[:]

        for i in range(len(self.history_s)):
            if self.broyden_type == "good":
                alpha = self._scalar_product(self.history_y[i], out)
            elif self.broyden_type == "bad":
                alpha = self._scalar_product(self.history_y[i], q)
            else:
                raise InputError(
                    "cashocs.space_mapping.optimal_control.SpaceMapping",
                    "broyden_type",
                    "broyden_type has to be either 'good' or 'bad'.",
                )

            for j in range(self.control_dim):
                out[j].vector()[:] += alpha * self.history_s[i][j].vector()[:]

    def _compute_bfgs_application(self, q, out):
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

    def _compute_ncg_direction(self, q, out):
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

    def _compute_eps(self):
        for i in range(self.control_dim):
            self.diff[i].vector()[:] = (
                self.p_current[i].vector()[:] - self.z_star[i].vector()[:]
            )
            eps = np.sqrt(self._scalar_product(self.diff, self.diff)) / self.norm_z_star

            return eps
