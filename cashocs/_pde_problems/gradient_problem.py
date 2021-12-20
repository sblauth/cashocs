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

"""Abstract implementation of a gradient problem.

This class is used to solve the Riesz projection to obtain
the gradient of the reduced cost functional.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import fenics
from petsc4py import PETSc

from .._interfaces.pde_problem import PDEProblem
from ..utils import _setup_petsc_options, _solve_linear_problem


if TYPE_CHECKING:
    from .._forms import ControlFormHandler
    from .state_problem import StateProblem
    from .adjoint_problem import AdjointProblem


class GradientProblem(PDEProblem):
    """A class representing the Riesz problem to determine the gradient."""

    def __init__(
        self,
        form_handler: ControlFormHandler,
        state_problem: StateProblem,
        adjoint_problem: AdjointProblem,
    ) -> None:
        """
        Parameters
        ----------
        form_handler : ControlFormHandler
            The FormHandler object of the optimization problem.
        state_problem : StateProblem
            The StateProblem object used to solve the state equations.
        adjoint_problem : AdjointProblem
            The AdjointProblem used to solve the adjoint equations.
        """

        super().__init__(form_handler)

        self.state_problem = state_problem
        self.adjoint_problem = adjoint_problem

        self.gradients = [fenics.Function(V) for V in self.form_handler.control_spaces]

        # Initialize the PETSc Krylov solver for the Riesz projection problems
        self.ksps = [PETSc.KSP().create() for i in range(self.form_handler.control_dim)]

        gradient_tol = self.config.getfloat(
            "OptimizationRoutine", "gradient_tol", fallback=1e-9
        )

        gradient_method = self.config.get(
            "OptimizationRoutine", "gradient_method", fallback="direct"
        )

        if gradient_method == "direct":
            option = [
                ["ksp_type", "preonly"],
                ["pc_type", "lu"],
                ["pc_factor_mat_solver_type", "mumps"],
                ["mat_mumps_icntl_24", 1],
            ]
        elif gradient_method == "iterative":
            option = [
                ["ksp_type", "cg"],
                ["pc_type", "hypre"],
                ["pc_hypre_type", "boomeramg"],
                ["pc_hypre_boomeramg_strong_threshold", 0.7],
                ["ksp_rtol", gradient_tol],
                ["ksp_atol", 1e-50],
                ["ksp_max_it", 250],
            ]

        self.riesz_ksp_options = []
        for i in range(self.form_handler.control_dim):
            self.riesz_ksp_options.append(option)

        _setup_petsc_options(self.ksps, self.riesz_ksp_options)
        for i, ksp in enumerate(self.ksps):
            ksp.setOperators(self.form_handler.riesz_projection_matrices[i])

        self.b_tensors = [
            fenics.PETScVector() for i in range(self.form_handler.control_dim)
        ]

    def solve(self) -> List[fenics.Function]:
        """Solves the Riesz projection problem to obtain the gradient of the (reduced) cost functional.

        Returns
        -------
        list[fenics.Function]
            The list of gradient of the cost functional.
        """

        self.state_problem.solve()
        self.adjoint_problem.solve()

        if not self.has_solution:
            for i in range(self.form_handler.control_dim):
                fenics.assemble(
                    self.form_handler.gradient_forms_rhs[i], tensor=self.b_tensors[i]
                )
                _solve_linear_problem(
                    ksp=self.ksps[i],
                    b=self.b_tensors[i].vec(),
                    x=self.gradients[i].vector().vec(),
                    ksp_options=self.riesz_ksp_options[i],
                )
                self.gradients[i].vector().apply("")

            self.has_solution = True

            self.gradient_norm_squared = self.form_handler.scalar_product(
                self.gradients, self.gradients
            )

            self.form_handler._post_hook()

        return self.gradients
