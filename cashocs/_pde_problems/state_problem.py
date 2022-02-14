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

"""Abstract implementation of a state equation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import fenics
from petsc4py import PETSc

from cashocs import nonlinear_solvers
from cashocs import utils
from cashocs._pde_problems import pde_problem

if TYPE_CHECKING:
    from cashocs import _forms


class StateProblem(pde_problem.PDEProblem):
    """The state system."""

    def __init__(
        self,
        form_handler: _forms.FormHandler,
        initial_guess: List[fenics.Function],
        temp_dict: Optional[Dict] = None,
    ) -> None:
        """Initializes self.

        Args:
            form_handler: The FormHandler of the optimization problem.
            initial_guess: An initial guess for the state variables, used to initialize
                them in each iteration.
            temp_dict: A dict used for reinitialization when remeshing is performed.
        """
        super().__init__(form_handler)

        self.initial_guess = initial_guess
        self.temp_dict = temp_dict

        self.bcs_list = self.form_handler.bcs_list
        self.states = self.form_handler.states

        self.picard_rtol = self.config.getfloat("StateSystem", "picard_rtol")
        self.picard_atol = self.config.getfloat("StateSystem", "picard_atol")
        self.picard_max_iter = self.config.getint("StateSystem", "picard_iter")
        self.picard_verbose = self.config.getboolean("StateSystem", "picard_verbose")
        self.newton_rtol = self.config.getfloat("StateSystem", "newton_rtol")
        self.newton_atol = self.config.getfloat("StateSystem", "newton_atol")
        self.newton_damped = self.config.getboolean("StateSystem", "newton_damped")
        self.newton_inexact = self.config.getboolean("StateSystem", "newton_inexact")
        self.newton_verbose = self.config.getboolean("StateSystem", "newton_verbose")
        self.newton_iter = self.config.getint("StateSystem", "newton_iter")

        self.newton_atols = [1] * self.form_handler.state_dim

        # noinspection PyUnresolvedReferences
        self.ksps = [PETSc.KSP().create() for _ in range(self.form_handler.state_dim)]
        utils._setup_petsc_options(self.ksps, self.form_handler.state_ksp_options)

        # adapt the tolerances so that the Newton system can be solved successfully
        if not self.form_handler.state_is_linear:
            for ksp in self.ksps:
                ksp.setTolerances(
                    rtol=self.newton_rtol / 100, atol=self.newton_atol / 100
                )

        self.A_tensors = [
            fenics.PETScMatrix() for _ in range(self.form_handler.state_dim)
        ]
        self.b_tensors = [
            fenics.PETScVector() for _ in range(self.form_handler.state_dim)
        ]
        self.res_j_tensors = [
            fenics.PETScVector() for _ in range(self.form_handler.state_dim)
        ]

        try:
            self.number_of_solves = self.temp_dict["output_dict"].get("state_solves", 0)
        except TypeError:
            self.number_of_solves = 0

    def _update_scalar_tracking_terms(self) -> None:
        """Updates the scalar_tracking_forms with current function values."""
        if self.form_handler.use_scalar_tracking:
            for j in range(self.form_handler.no_scalar_tracking_terms):
                scalar_integral_value = fenics.assemble(
                    self.form_handler.scalar_cost_functional_integrands[j]
                )
                self.form_handler.scalar_cost_functional_integrand_values[
                    j
                ].vector().vec().set(scalar_integral_value)

    def _update_min_max_terms(self) -> None:
        """Updates the min_max_terms with current function values."""
        if self.form_handler.use_min_max_terms:
            for j in range(self.form_handler.no_min_max_terms):
                min_max_integral_value = fenics.assemble(
                    self.form_handler.min_max_integrands[j]
                )
                self.form_handler.min_max_integrand_values[j].vector().vec().set(
                    min_max_integral_value
                )

    def solve(self) -> List[fenics.Function]:
        """Solves the state system.

        Returns:
            The solution of the state system.
        """
        if not self.has_solution:

            self.form_handler._pre_hook()

            if self.initial_guess is not None:
                for j in range(self.form_handler.state_dim):
                    fenics.assign(self.states[j], self.initial_guess[j])

            if (
                not self.form_handler.state_is_picard
                or self.form_handler.state_dim == 1
            ):
                if self.form_handler.state_is_linear:
                    for i in range(self.form_handler.state_dim):
                        utils._assemble_and_solve_linear(
                            self.form_handler.state_eq_forms_lhs[i],
                            self.form_handler.state_eq_forms_rhs[i],
                            self.bcs_list[i],
                            A=self.A_tensors[i],
                            b=self.b_tensors[i],
                            x=self.states[i].vector().vec(),
                            ksp=self.ksps[i],
                            ksp_options=self.form_handler.state_ksp_options[i],
                        )
                        self.states[i].vector().apply("")

                else:
                    for i in range(self.form_handler.state_dim):
                        self.states[i] = nonlinear_solvers.newton_solve(
                            self.form_handler.state_eq_forms[i],
                            self.states[i],
                            self.bcs_list[i],
                            rtol=self.newton_rtol,
                            atol=self.newton_atol,
                            max_iter=self.newton_iter,
                            damped=self.newton_damped,
                            inexact=self.newton_inexact,
                            verbose=self.newton_verbose,
                            ksp=self.ksps[i],
                            ksp_options=self.form_handler.state_ksp_options[i],
                            A_tensor=self.A_tensors[i],
                            b_tensor=self.b_tensors[i],
                        )

            else:
                nonlinear_solvers.picard_iteration(
                    self.form_handler.state_eq_forms,
                    self.states,
                    self.bcs_list,
                    max_iter=self.picard_max_iter,
                    rtol=self.picard_rtol,
                    atol=self.picard_atol,
                    verbose=self.picard_verbose,
                    inner_damped=self.newton_damped,
                    inner_inexact=self.newton_inexact,
                    inner_verbose=self.newton_verbose,
                    inner_max_its=self.newton_iter,
                    ksps=self.ksps,
                    ksp_options=self.form_handler.state_ksp_options,
                    A_tensors=self.A_tensors,
                    b_tensors=self.b_tensors,
                    inner_is_linear=self.form_handler.state_is_linear,
                )

            self.has_solution = True
            self.number_of_solves += 1

            self._update_scalar_tracking_terms()
            self._update_min_max_terms()

        return self.states
