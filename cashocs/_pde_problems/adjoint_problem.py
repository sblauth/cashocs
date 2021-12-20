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

"""Abstract implementation of an adjoint problem.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import fenics
import numpy as np
from petsc4py import PETSc

from .._exceptions import NotConvergedError
from .._interfaces import PDEProblem
from ..utils import _assemble_petsc_system, _setup_petsc_options, _solve_linear_problem


if TYPE_CHECKING:
    from .._forms import FormHandler
    from .state_problem import StateProblem


class AdjointProblem(PDEProblem):
    """The adjoint problem.

    This class implements the adjoint problem as well as its solver.
    """

    def __init__(
        self,
        form_handler: FormHandler,
        state_problem: StateProblem,
        temp_dict: Dict = None,
    ) -> None:
        """
        Parameters
        ----------
        form_handler : FormHandler
            The FormHandler object for the optimization problem.
        state_problem : StateProblem
            The StateProblem object used to get the point where we linearize the problem.
        temp_dict : dict
            A dictionary used for reinitializations when remeshing is performed.
        """

        super().__init__(form_handler)

        self.state_problem = state_problem
        self.temp_dict = temp_dict

        self.adjoints = self.form_handler.adjoints
        self.bcs_list_ad = self.form_handler.bcs_list_ad

        self.rtol = self.config.getfloat("StateSystem", "picard_rtol", fallback=1e-10)
        self.atol = self.config.getfloat("StateSystem", "picard_atol", fallback=1e-12)
        self.maxiter = self.config.getint("StateSystem", "picard_iter", fallback=50)
        self.picard_verbose = self.config.getboolean(
            "StateSystem", "picard_verbose", fallback=False
        )

        self.ksps = [PETSc.KSP().create() for i in range(self.form_handler.state_dim)]
        _setup_petsc_options(self.ksps, self.form_handler.adjoint_ksp_options)

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
            self.number_of_solves = self.temp_dict["output_dict"].get(
                "adjoint_solves", 0
            )
        except TypeError:
            self.number_of_solves = 0

    def solve(self) -> List[fenics.Function]:
        """Solves the adjoint system.

        Returns
        -------
        list[fenics.Function]
            The list of adjoint variables.
        """

        self.state_problem.solve()

        if not self.has_solution:
            if (
                not self.form_handler.state_is_picard
                or self.form_handler.state_dim == 1
            ):
                for i in range(self.form_handler.state_dim):
                    _assemble_petsc_system(
                        self.form_handler.adjoint_eq_lhs[-1 - i],
                        self.form_handler.adjoint_eq_rhs[-1 - i],
                        self.bcs_list_ad[-1 - i],
                        A_tensor=self.A_tensors[-1 - i],
                        b_tensor=self.b_tensors[-1 - i],
                    )
                    _solve_linear_problem(
                        self.ksps[-1 - i],
                        self.A_tensors[-1 - i].mat(),
                        self.b_tensors[-1 - i].vec(),
                        self.adjoints[-1 - i].vector().vec(),
                        self.form_handler.adjoint_ksp_options[-1 - i],
                    )
                    self.adjoints[-1 - i].vector().apply("")

            else:
                for i in range(self.maxiter + 1):
                    res = 0.0
                    for j in range(self.form_handler.state_dim):
                        fenics.assemble(
                            self.form_handler.adjoint_picard_forms[j],
                            tensor=self.res_j_tensors[j],
                        )
                        [
                            bc.apply(self.res_j_tensors[j])
                            for bc in self.form_handler.bcs_list_ad[j]
                        ]
                        res += pow(self.res_j_tensors[j].norm("l2"), 2)

                    if res == 0:
                        break

                    res = np.sqrt(res)

                    if i == 0:
                        res_0 = res

                    if self.picard_verbose:
                        print(
                            f"Iteration {i:d}: ||res|| (abs) = {res:.3e}   ||res|| (rel) = {res/res_0:.3e}"
                        )

                    if res / res_0 < self.rtol or res < self.atol:
                        break

                    if i == self.maxiter:
                        raise NotConvergedError(
                            "Picard iteration for the adjoint system"
                        )

                    for j in range(self.form_handler.state_dim):
                        _assemble_petsc_system(
                            self.form_handler.adjoint_eq_lhs[-1 - j],
                            self.form_handler.adjoint_eq_rhs[-1 - j],
                            self.bcs_list_ad[-1 - j],
                            A_tensor=self.A_tensors[-1 - j],
                            b_tensor=self.b_tensors[-1 - j],
                        )
                        _solve_linear_problem(
                            self.ksps[-1 - j],
                            self.A_tensors[-1 - j].mat(),
                            self.b_tensors[-1 - j].vec(),
                            self.adjoints[-1 - j].vector().vec(),
                            self.form_handler.adjoint_ksp_options[-1 - j],
                        )
                        self.adjoints[-1 - j].vector().apply("")

            if self.picard_verbose and self.form_handler.state_is_picard:
                print("")
            self.has_solution = True
            self.number_of_solves += 1

        return self.adjoints
