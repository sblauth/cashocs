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

"""Abstract implementation of a state equation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import fenics
import numpy as np
from petsc4py import PETSc

from .._exceptions import NotConvergedError
from .._interfaces import PDEProblem
from ..nonlinear_solvers import newton_solve
from ..utils import _assemble_petsc_system, _setup_petsc_options, _solve_linear_problem


if TYPE_CHECKING:
    from .._forms import FormHandler


class StateProblem(PDEProblem):
    """The state system."""

    def __init__(
        self,
        form_handler: FormHandler,
        initial_guess: List[fenics.Function],
        temp_dict: Optional[Dict] = None,
    ) -> None:
        """Initializes the state system.

        Parameters
        ----------
        form_handler : FormHandler
            The FormHandler of the optimization problem.
        initial_guess : list[fenics.Function]
            An initial guess for the state variables, used to initialize them in each iteration.
        temp_dict : dict or None, optional
            A dict used for reinitialization when remeshing is performed.
        """

        super().__init__(form_handler)

        self.initial_guess = initial_guess
        self.temp_dict = temp_dict

        self.bcs_list = self.form_handler.bcs_list
        self.states = self.form_handler.states

        self.rtol = self.config.getfloat("StateSystem", "picard_rtol", fallback=1e-10)
        self.atol = self.config.getfloat("StateSystem", "picard_atol", fallback=1e-20)
        self.maxiter = self.config.getint("StateSystem", "picard_iter", fallback=50)
        self.picard_verbose = self.config.getboolean(
            "StateSystem", "picard_verbose", fallback=False
        )
        self.newton_rtol = self.config.getfloat(
            "StateSystem", "newton_rtol", fallback=1e-11
        )
        self.newton_atol = self.config.getfloat(
            "StateSystem", "newton_atol", fallback=1e-13
        )
        self.newton_damped = self.config.getboolean(
            "StateSystem", "newton_damped", fallback=True
        )
        self.newton_inexact = self.config.getboolean(
            "StateSystem", "newton_inexact", fallback=False
        )
        self.newton_verbose = self.config.getboolean(
            "StateSystem", "newton_verbose", fallback=False
        )
        self.newton_iter = self.config.getint("StateSystem", "newton_iter", fallback=50)

        self.newton_atols = [1 for i in range(self.form_handler.state_dim)]

        self.ksps = [PETSc.KSP().create() for i in range(self.form_handler.state_dim)]
        _setup_petsc_options(self.ksps, self.form_handler.state_ksp_options)

        # adapt the tolerances so that the Newton system can be solved sucessfully
        if not self.form_handler.state_is_linear:
            for ksp in self.ksps:
                ksp.setTolerances(
                    rtol=self.newton_rtol / 100, atol=self.newton_atol / 100
                )

        self.A_tensors = [
            fenics.PETScMatrix() for i in range(self.form_handler.state_dim)
        ]
        self.b_tensors = [
            fenics.PETScVector() for i in range(self.form_handler.state_dim)
        ]
        self.res_j_tensors = [
            fenics.PETScVector() for i in range(self.form_handler.state_dim)
        ]

        try:
            self.number_of_solves = self.temp_dict["output_dict"].get("state_solves", 0)
        except TypeError:
            self.number_of_solves = 0

    def solve(self) -> List[fenics.Function]:
        """Solves the state system.

        Returns
        -------
        list[fenics.Function]
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
                        _assemble_petsc_system(
                            self.form_handler.state_eq_forms_lhs[i],
                            self.form_handler.state_eq_forms_rhs[i],
                            self.bcs_list[i],
                            A_tensor=self.A_tensors[i],
                            b_tensor=self.b_tensors[i],
                        )
                        _solve_linear_problem(
                            self.ksps[i],
                            self.A_tensors[i].mat(),
                            self.b_tensors[i].vec(),
                            self.states[i].vector().vec(),
                            self.form_handler.state_ksp_options[i],
                        )
                        self.states[i].vector().apply("")

                else:
                    for i in range(self.form_handler.state_dim):
                        if self.initial_guess is not None:
                            fenics.assign(self.states[i], self.initial_guess[i])

                        self.states[i] = newton_solve(
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
                for i in range(self.maxiter + 1):
                    res = 0.0
                    for j in range(self.form_handler.state_dim):
                        fenics.assemble(
                            self.form_handler.state_picard_forms[j],
                            tensor=self.res_j_tensors[j],
                        )

                        [
                            bc.apply(self.res_j_tensors[j])
                            for bc in self.form_handler.bcs_list_ad[j]
                        ]

                        if self.number_of_solves == 0 and i == 0:
                            self.newton_atols[j] = (
                                self.res_j_tensors[j].norm("l2") * self.newton_atol
                            )
                            if self.res_j_tensors[j].norm("l2") == 0.0:
                                self.newton_atols[j] = self.newton_atol

                        res += pow(self.res_j_tensors[j].norm("l2"), 2)

                    if res == 0:
                        break
                    res = np.sqrt(res)
                    if i == 0:
                        res_0 = res
                    if self.picard_verbose:
                        print(
                            f"Iteration {i:d}: ||res|| (abs): {res:.3e}   ||res|| (rel): {res/res_0:.3e}"
                        )
                    if res / res_0 < self.rtol or res < self.atol:
                        break

                    if i == self.maxiter:
                        raise NotConvergedError("Picard iteration for the state system")

                    for j in range(self.form_handler.state_dim):
                        if self.initial_guess is not None:
                            fenics.assign(self.states[j], self.initial_guess[j])

                        # adapt tolerances so that a solution is possible
                        if not self.form_handler.state_is_linear:
                            self.ksps[j].setTolerances(
                                rtol=np.minimum(0.9 * res, 0.9) / 100,
                                atol=self.newton_atols[j] / 100,
                            )

                            self.states[j] = newton_solve(
                                self.form_handler.state_eq_forms[j],
                                self.states[j],
                                self.bcs_list[j],
                                rtol=np.minimum(0.9 * res, 0.9),
                                atol=self.newton_atols[j],
                                max_iter=self.newton_iter,
                                damped=self.newton_damped,
                                verbose=self.newton_verbose,
                                ksp=self.ksps[j],
                                ksp_options=self.form_handler.state_ksp_options[j],
                                A_tensor=self.A_tensors[j],
                                b_tensor=self.b_tensors[j],
                            )
                        else:
                            _assemble_petsc_system(
                                self.form_handler.state_eq_forms_lhs[j],
                                self.form_handler.state_eq_forms_rhs[j],
                                self.bcs_list[j],
                                A_tensor=self.A_tensors[j],
                                b_tensor=self.b_tensors[j],
                            )
                            _solve_linear_problem(
                                self.ksps[j],
                                self.A_tensors[j].mat(),
                                self.b_tensors[j].vec(),
                                self.states[j].vector().vec(),
                                self.form_handler.state_ksp_options[j],
                            )
                            self.states[j].vector().apply("")

            if self.picard_verbose and self.form_handler.state_is_picard:
                print("")
            self.has_solution = True
            self.number_of_solves += 1

            if self.form_handler.use_scalar_tracking:
                for j in range(self.form_handler.no_scalar_tracking_terms):
                    scalar_integral_value = fenics.assemble(
                        self.form_handler.scalar_cost_functional_integrands[j]
                    )
                    self.form_handler.scalar_cost_functional_integrand_values[
                        j
                    ].vector().vec().set(scalar_integral_value)

            if self.form_handler.use_min_max_terms:
                for j in range(self.form_handler.no_min_max_terms):
                    min_max_integral_value = fenics.assemble(
                        self.form_handler.min_max_integrands[j]
                    )
                    self.form_handler.min_max_integrand_values[j].vector().vec().set(
                        min_max_integral_value
                    )

        return self.states
