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

"""Adjoint system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import ufl

from cashocs import _utils
from cashocs import log
from cashocs import nonlinear_solvers
from cashocs._pde_problems import pde_problem

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs._database import database
    from cashocs._pde_problems import state_problem as sp


class AdjointProblem(pde_problem.PDEProblem):
    """This class implements the adjoint problem as well as its solver."""

    def __init__(
        self,
        db: database.Database,
        adjoint_form_handler: _forms.AdjointFormHandler,
        state_problem: sp.StateProblem,
        linear_solver: _utils.linalg.LinearSolver | None = None,
        adjoint_linearizations: list[ufl.Form] | None = None,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            adjoint_form_handler: The form handler for the adjoint system.
            state_problem: The StateProblem object used to get the point where we
                linearize the problem.
            linear_solver: The linear solver (KSP) which is used to solve the linear
                systems arising from the discretized PDE.
            adjoint_linearizations: The UFL form of the linearization to be employed for
                solving the adjoint system. Defaults to None.

        """
        super().__init__(db, linear_solver=linear_solver)

        self.adjoint_form_handler = adjoint_form_handler
        self.state_problem = state_problem

        self.excluded_from_time_derivative = (
            self.state_problem.excluded_from_time_derivative
        )
        if adjoint_linearizations is not None:
            self.adjoint_linearizations = adjoint_linearizations
        else:
            self.adjoint_linearizations = [None] * self.db.parameter_db.state_dim

        self.adjoints = self.db.function_db.adjoints
        self.bcs_list_ad = self.adjoint_form_handler.bcs_list_ad

        self.picard_rtol: float = self.config.getfloat("StateSystem", "picard_rtol")
        self.picard_atol: float = self.config.getfloat("StateSystem", "picard_atol")
        self.picard_max_iter: int = self.config.getint("StateSystem", "picard_iter")
        self.picard_verbose: bool = self.config.getboolean(
            "StateSystem", "picard_verbose"
        )

        # pylint: disable=invalid-name
        self.A_tensors = [
            fenics.PETScMatrix(db.geometry_db.mpi_comm)
            for _ in range(self.db.parameter_db.state_dim)
        ]
        self.b_tensors = [
            fenics.PETScVector(db.geometry_db.mpi_comm)
            for _ in range(self.db.parameter_db.state_dim)
        ]

        self.res_j_tensors = [
            fenics.PETScVector(db.geometry_db.mpi_comm)
            for _ in range(self.db.parameter_db.state_dim)
        ]

        self._number_of_solves = 0
        if self.db.parameter_db.temp_dict:
            if (
                "no_adjoint_solves"
                in self.db.parameter_db.temp_dict["output_dict"].keys()
            ):
                self.number_of_solves: int = self.db.parameter_db.temp_dict[
                    "output_dict"
                ]["no_adjoint_solves"][-1]
            else:
                self.number_of_solves = 0
        else:
            self.number_of_solves = 0

    @property
    def number_of_solves(self) -> int:
        """Counts the number of solves of the adjoint problem."""
        return self._number_of_solves

    @number_of_solves.setter
    def number_of_solves(self, value: int) -> None:
        self.db.parameter_db.optimization_state["no_adjoint_solves"] = value
        self._number_of_solves = value

    def solve(self) -> list[fenics.Function]:
        """Solves the adjoint system.

        Returns:
            The list of adjoint variables.

        """
        self.state_problem.solve()

        if not self.has_solution:
            log.begin("Solving the adjoint system.", level=log.DEBUG)
            if (
                not self.config.getboolean("StateSystem", "picard_iteration")
                or self.db.parameter_db.state_dim == 1
            ):
                for i in range(self.db.parameter_db.state_dim):
                    eftd = self.excluded_from_time_derivative[-1 - i]
                    if "ts" in _utils.get_petsc_prefixes(
                        self.db.parameter_db.adjoint_ksp_options[-1 - i]
                    ):
                        nonlinear_solvers.ts_pseudo_solve(
                            self.adjoint_form_handler.adjoint_eq_forms[-1 - i],
                            self.adjoints[-1 - i],
                            self.bcs_list_ad[-1 - i],
                            derivative=self.adjoint_linearizations[-1 - i],
                            petsc_options=self.db.parameter_db.adjoint_ksp_options[
                                -1 - i
                            ],
                            A_tensor=self.A_tensors[-1 - i],
                            b_tensor=self.b_tensors[-1 - i],
                            preconditioner_form=self.db.form_db.preconditioner_forms[
                                -1 - i
                            ],
                            excluded_from_time_derivative=eftd,
                        )
                    elif self.db.config.getboolean(
                        "StateSystem", "use_adjoint_linearizations"
                    ):
                        nonlinear_solvers.snes_solve(
                            self.adjoint_form_handler.adjoint_eq_forms[-1 - i],
                            self.adjoints[-1 - i],
                            self.bcs_list_ad[-1 - i],
                            derivative=self.adjoint_linearizations[-1 - i],
                            petsc_options=self.db.parameter_db.adjoint_ksp_options[
                                -1 - i
                            ],
                            A_tensor=self.A_tensors[-1 - i],
                            b_tensor=self.b_tensors[-1 - i],
                            preconditioner_form=self.db.form_db.preconditioner_forms[
                                -1 - i
                            ],
                        )
                    else:
                        _utils.assemble_and_solve_linear(
                            self.adjoint_form_handler.adjoint_eq_lhs[-1 - i],
                            self.adjoint_form_handler.adjoint_eq_rhs[-1 - i],
                            self.adjoints[-1 - i],
                            bcs=self.bcs_list_ad[-1 - i],
                            A=self.A_tensors[-1 - i],
                            b=self.b_tensors[-1 - i],
                            ksp_options=self.db.parameter_db.adjoint_ksp_options[
                                -1 - i
                            ],
                            preconditioner_form=self.db.form_db.preconditioner_forms[
                                -1 - i
                            ],
                            linear_solver=self.linear_solver,
                        )

            else:
                nonlinear_solvers.picard_iteration(
                    self.adjoint_form_handler.adjoint_eq_forms[::-1],
                    self.adjoints[::-1],
                    self.bcs_list_ad[::-1],
                    max_iter=self.picard_max_iter,
                    rtol=self.picard_rtol,
                    atol=self.picard_atol,
                    verbose=self.picard_verbose,
                    inner_max_iter=2,
                    ksp_options=self.db.parameter_db.adjoint_ksp_options[::-1],
                    A_tensors=self.A_tensors[::-1],
                    b_tensors=self.b_tensors[::-1],
                    preconditioner_forms=self.db.form_db.preconditioner_forms[::-1],
                    newton_linearizations=self.adjoint_linearizations[::-1],
                )

            self.has_solution = True
            self.number_of_solves += 1
            log.end()

        return self.adjoints
