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

"""Gradient problem.

This class is used to solve the Riesz projection to obtain the gradient of the reduced
cost functional.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import fenics

from cashocs import _forms
from cashocs import _utils
from cashocs import log
from cashocs._pde_problems import pde_problem

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._pde_problems import adjoint_problem as ap
    from cashocs._pde_problems import state_problem as sp


class ControlGradientProblem(pde_problem.PDEProblem):
    """A class representing the Riesz problem to determine the gradient."""

    def __init__(
        self,
        db: database.Database,
        form_handler: _forms.ControlFormHandler,
        state_problem: sp.StateProblem,
        adjoint_problem: ap.AdjointProblem,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The FormHandler object of the optimization problem.
            state_problem: The StateProblem object used to solve the state equations.
            adjoint_problem: The AdjointProblem used to solve the adjoint equations.

        """
        super().__init__(db)

        self.form_handler = form_handler
        self.state_problem = state_problem
        self.adjoint_problem = adjoint_problem

        self.gradient_norm_squared = 1.0

        gradient_tol: float = self.config.getfloat(
            "OptimizationRoutine", "gradient_tol"
        )

        gradient_method: str = self.config.get("OptimizationRoutine", "gradient_method")

        if db.parameter_db.gradient_ksp_options is not None:
            self.riesz_ksp_options = db.parameter_db.gradient_ksp_options
        else:
            if gradient_method.casefold() == "direct":
                option: _typing.KspOption = copy.deepcopy(
                    _utils.linalg.direct_ksp_options
                )
            elif gradient_method.casefold() == "iterative":
                option = copy.deepcopy(_utils.linalg.iterative_ksp_options)
                option["ksp_rtol"] = gradient_tol

            self.riesz_ksp_options = []
            for _ in range(len(self.db.function_db.gradient)):
                self.riesz_ksp_options.append(option)

        self.b_tensors = [
            fenics.PETScVector(self.db.geometry_db.mpi_comm)
            for _ in range(len(self.db.function_db.gradient))
        ]

    def solve(self) -> list[fenics.Function]:
        """Solves the Riesz projection problem to obtain the gradient.

        Returns:
            The list containing the (components of the) gradient of the cost functional.

        """
        self.state_problem.solve()
        self.adjoint_problem.solve()

        if not self.has_solution:
            log.begin("Computing the gradient.", level=log.DEBUG)
            for i in range(len(self.db.function_db.gradient)):
                self.form_handler.assemblers[i].assemble(self.b_tensors[i])
                self.linear_solver.solve(
                    self.db.function_db.gradient[i],
                    A=self.form_handler.riesz_projection_matrices[i],
                    b=self.b_tensors[i].vec(),
                    ksp_options=self.riesz_ksp_options[i],
                )

            self.has_solution = True

            self.gradient_norm_squared = self.form_handler.scalar_product(
                self.db.function_db.gradient, self.db.function_db.gradient
            )

            self.db.callback.call_post()
            log.end()

        return self.db.function_db.gradient
