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

"""Abstract implementation of an adjoint problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import fenics
from petsc4py import PETSc

from cashocs import nonlinear_solvers
from cashocs import utils
from cashocs._pde_problems import pde_problem

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs._pde_problems import state_problem as sp


class AdjointProblem(pde_problem.PDEProblem):
    """This class implements the adjoint problem as well as its solver."""

    def __init__(
        self,
        form_handler: _forms.FormHandler,
        state_problem: sp.StateProblem,
        temp_dict: Dict = None,
    ) -> None:
        """
        Args:
            form_handler: The FormHandler object for the optimization problem.
            state_problem: The StateProblem object used to get the point where we
                linearize the problem.
            temp_dict: A dictionary used for reinitializations when remeshing is
                performed.
        """

        super().__init__(form_handler)

        self.state_problem = state_problem
        self.temp_dict = temp_dict

        self.adjoints = self.form_handler.adjoints
        self.bcs_list_ad = self.form_handler.bcs_list_ad

        self.picard_rtol = self.config.getfloat(
            "StateSystem", "picard_rtol", fallback=1e-10
        )
        self.picard_atol = self.config.getfloat(
            "StateSystem", "picard_atol", fallback=1e-12
        )
        self.picard_max_iter = self.config.getint(
            "StateSystem", "picard_iter", fallback=50
        )
        self.picard_verbose = self.config.getboolean(
            "StateSystem", "picard_verbose", fallback=False
        )

        # noinspection PyUnresolvedReferences
        self.ksps = [PETSc.KSP().create() for _ in range(self.form_handler.state_dim)]
        utils._setup_petsc_options(self.ksps, self.form_handler.adjoint_ksp_options)

        self.A_tensors = [
            fenics.PETScMatrix() for _ in range(self.form_handler.state_dim)
        ]
        self.b_tensors = [
            fenics.PETScVector() for _ in range(self.form_handler.state_dim)
        ]

        self.res_j_tensors = [
            fenics.PETScVector() for _ in range(self.form_handler.state_dim)
        ]

        if self.form_handler.is_shape_problem and self.temp_dict is not None:
            self.number_of_solves = self.temp_dict["output_dict"].get(
                "adjoint_solves", 0
            )
        else:
            self.number_of_solves = 0

    def solve(self) -> List[fenics.Function]:
        """Solves the adjoint system.

        Returns:
            The list of adjoint variables.
        """

        self.state_problem.solve()

        if not self.has_solution:
            if (
                not self.form_handler.state_is_picard
                or self.form_handler.state_dim == 1
            ):
                for i in range(self.form_handler.state_dim):
                    utils._assemble_and_solve_linear(
                        self.form_handler.adjoint_eq_lhs[-1 - i],
                        self.form_handler.adjoint_eq_rhs[-1 - i],
                        self.bcs_list_ad[-1 - i],
                        A=self.A_tensors[-1 - i],
                        b=self.b_tensors[-1 - i],
                        x=self.adjoints[-1 - i].vector().vec(),
                        ksp=self.ksps[-1 - i],
                        ksp_options=self.form_handler.adjoint_ksp_options[-1 - i],
                    )
                    self.adjoints[-1 - i].vector().apply("")

            else:
                nonlinear_solvers.picard_iteration(
                    self.form_handler.adjoint_eq_forms,
                    self.adjoints,
                    self.bcs_list_ad,
                    max_iter=self.picard_max_iter,
                    rtol=self.picard_rtol,
                    atol=self.picard_atol,
                    verbose=self.picard_verbose,
                    inner_damped=False,
                    inner_inexact=False,
                    inner_verbose=False,
                    inner_max_its=2,
                    ksps=self.ksps,
                    ksp_options=self.form_handler.adjoint_ksp_options,
                    A_tensors=self.A_tensors,
                    b_tensors=self.b_tensors,
                    inner_is_linear=True,
                )

            self.has_solution = True
            self.number_of_solves += 1

        return self.adjoints
