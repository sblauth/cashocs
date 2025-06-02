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

"""State system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics

from cashocs import _utils
from cashocs import log
from cashocs import nonlinear_solvers
from cashocs._pde_problems import pde_problem

if TYPE_CHECKING:
    try:
        import ufl_legacy as ufl
    except ImportError:
        import ufl
    from cashocs import _forms
    from cashocs._database import database


class StateProblem(pde_problem.PDEProblem):
    """The state system."""

    def __init__(
        self,
        db: database.Database,
        state_form_handler: _forms.StateFormHandler,
        initial_guess: list[fenics.Function] | None,
        linear_solver: _utils.linalg.LinearSolver | None = None,
        newton_linearizations: list[ufl.Form | None] | None = None,
        excluded_from_time_derivative: list[list[int]] | list[None] | None = None,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            state_form_handler: The form handler for the state problem.
            initial_guess: An initial guess for the state variables, used to initialize
                them in each iteration.
            linear_solver: The linear solver (KSP) which is used to solve the linear
                systems arising from the discretized PDE.
            newton_linearizations: A (list of) UFL forms describing which (alternative)
                linearizations should be used for the (nonlinear) state equations when
                solving them (with Newton's method). The default is `None`, so that the
                Jacobian of the supplied state forms is used.
            excluded_from_time_derivative: For each state equation, a list of indices
                which are not part of the first order time derivative for pseudo time
                stepping. Example: Pressure for incompressible flow. Default is None.

        """
        super().__init__(db, linear_solver=linear_solver)

        self.state_form_handler = state_form_handler
        self.initial_guess = initial_guess
        if newton_linearizations is not None:
            self.newton_linearizations = newton_linearizations
        else:
            self.newton_linearizations = [None] * self.db.parameter_db.state_dim

        if excluded_from_time_derivative is not None:
            self.excluded_from_time_derivative = excluded_from_time_derivative
        else:
            self.excluded_from_time_derivative = [None] * self.db.parameter_db.state_dim

        self.bcs_list: list[list[fenics.DirichletBC]] = self.state_form_handler.bcs_list
        self.states = self.db.function_db.states
        self.states_checkpoint = [fun.copy(True) for fun in self.states]

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
        self.backend = self.config.get("StateSystem", "backend")

        # pylint: disable=invalid-name
        self.A_tensors = [
            fenics.PETScMatrix(self.db.geometry_db.mpi_comm)
            for _ in range(self.db.parameter_db.state_dim)
        ]
        self.b_tensors = [
            fenics.PETScVector(self.db.geometry_db.mpi_comm)
            for _ in range(self.db.parameter_db.state_dim)
        ]
        self.res_j_tensors = [
            fenics.PETScVector(self.db.geometry_db.mpi_comm)
            for _ in range(self.db.parameter_db.state_dim)
        ]

        self._number_of_solves = 0
        if self.db.parameter_db.temp_dict:
            if (
                "no_state_solves"
                in self.db.parameter_db.temp_dict["output_dict"].keys()
            ):
                self.number_of_solves: int = self.db.parameter_db.temp_dict[
                    "output_dict"
                ]["no_state_solves"][-1]
            else:
                self.number_of_solves = 0
        else:
            self.number_of_solves = 0

    @property
    def number_of_solves(self) -> int:
        """Counts the number of solves of the state problem."""
        return self._number_of_solves

    @number_of_solves.setter
    def number_of_solves(self, value: int) -> None:
        self.db.parameter_db.optimization_state["no_state_solves"] = value
        self._number_of_solves = value

    def _update_cost_functionals(self) -> None:
        for functional in self.db.form_db.cost_functional_list:
            functional.update()

    def solve(self) -> list[fenics.Function]:
        """Solves the state system.

        Returns:
            The solution of the state system.

        """
        if not self.has_solution:
            log.begin("Solving the state system.", level=log.DEBUG)

            self.db.callback.call_pre()
            self._generate_checkpoint()
            if (
                not self.config.getboolean("StateSystem", "picard_iteration")
                or self.db.parameter_db.state_dim == 1
            ):
                if self.config.getboolean("StateSystem", "is_linear"):
                    for i in range(self.db.parameter_db.state_dim):
                        if self.initial_guess is not None:
                            fenics.assign(self.states[i], self.initial_guess[i])
                        _utils.assemble_and_solve_linear(
                            self.state_form_handler.state_eq_forms_lhs[i],
                            self.state_form_handler.state_eq_forms_rhs[i],
                            self.states[i],
                            bcs=self.bcs_list[i],
                            A=self.A_tensors[i],
                            b=self.b_tensors[i],
                            ksp_options=self.db.parameter_db.state_ksp_options[i],
                            preconditioner_form=self.db.form_db.preconditioner_forms[i],
                            linear_solver=self.linear_solver,
                        )

                else:
                    for i in range(self.db.parameter_db.state_dim):
                        if self.initial_guess is not None:
                            fenics.assign(self.states[i], self.initial_guess[i])

                        pc_forms = self.db.form_db.preconditioner_forms[i]
                        petsc_options = self.db.parameter_db.state_ksp_options[i]
                        eftd = self.excluded_from_time_derivative[i]
                        if self.backend == "petsc":
                            if "ts" in _utils.get_petsc_prefixes(
                                self.db.parameter_db.state_ksp_options[i]
                            ):
                                nonlinear_solvers.ts_pseudo_solve(
                                    self.state_form_handler.state_eq_forms[i],
                                    self.states[i],
                                    self.bcs_list[i],
                                    derivative=self.newton_linearizations[i],
                                    petsc_options=petsc_options,
                                    A_tensor=self.A_tensors[i],
                                    b_tensor=self.b_tensors[i],
                                    preconditioner_form=pc_forms,
                                    excluded_from_time_derivative=eftd,
                                )
                            else:
                                nonlinear_solvers.snes_solve(
                                    self.state_form_handler.state_eq_forms[i],
                                    self.states[i],
                                    self.bcs_list[i],
                                    derivative=self.newton_linearizations[i],
                                    petsc_options=petsc_options,
                                    A_tensor=self.A_tensors[i],
                                    b_tensor=self.b_tensors[i],
                                    preconditioner_form=pc_forms,
                                )
                        else:
                            nonlinear_solvers.newton_solve(
                                self.state_form_handler.state_eq_forms[i],
                                self.states[i],
                                self.bcs_list[i],
                                derivative=self.newton_linearizations[i],
                                rtol=self.newton_rtol,
                                atol=self.newton_atol,
                                max_iter=self.newton_iter,
                                damped=self.newton_damped,
                                inexact=self.newton_inexact,
                                verbose=self.newton_verbose,
                                ksp_options=self.db.parameter_db.state_ksp_options[i],
                                A_tensor=self.A_tensors[i],
                                b_tensor=self.b_tensors[i],
                                preconditioner_form=pc_forms,
                                linear_solver=self.linear_solver,
                            )

            else:
                nonlinear_solvers.picard_iteration(
                    self.state_form_handler.state_eq_forms,
                    self.states,
                    self.bcs_list,
                    max_iter=self.picard_max_iter,
                    rtol=self.picard_rtol,
                    atol=self.picard_atol,
                    verbose=self.picard_verbose,
                    inner_max_iter=self.newton_iter,
                    ksp_options=self.db.parameter_db.state_ksp_options,
                    A_tensors=self.A_tensors,
                    b_tensors=self.b_tensors,
                    preconditioner_forms=self.db.form_db.preconditioner_forms,
                    newton_linearizations=self.newton_linearizations,
                )

            self.has_solution = True
            self.number_of_solves += 1
            log.end()

            self._update_cost_functionals()

        return self.states

    def _generate_checkpoint(self) -> None:
        """Generates a checkpoint of the state variables."""
        for i in range(len(self.states)):
            self.states_checkpoint[i].vector().vec().aypx(
                0.0, self.states[i].vector().vec()
            )
            self.states_checkpoint[i].vector().apply("")

    def revert_to_checkpoint(self) -> None:
        """Reverts the state variables to a checkpointed value.

        This is useful when the solution of the state problem fails and another attempt
        is made to solve it. Then, the perturbed solution of Newton's method should not
        be the initial guess.

        """
        for i in range(len(self.states)):
            self.states[i].vector().vec().aypx(
                0.0, self.states_checkpoint[i].vector().vec()
            )
            self.states[i].vector().apply("")

        log.end()
