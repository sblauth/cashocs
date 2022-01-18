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

"""Picard iteration for coupled PDEs."""

from __future__ import annotations

from typing import List, Union, Optional

import fenics
import numpy as np
import ufl
from petsc4py import PETSc

from cashocs import _exceptions
from cashocs import utils
from cashocs.nonlinear_solvers import newton_solver


# noinspection PyPep8Naming
def picard_iteration(
    F_list: Union[List[ufl.form], ufl.Form],
    u_list: Union[List[fenics.Function], fenics.Function],
    bcs_list: Union[List[fenics.DirichletBC], List[List[fenics.DirichletBC]]],
    max_iter: int = 50,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    verbose: bool = True,
    inner_damped: bool = True,
    inner_inexact: bool = True,
    inner_verbose: bool = False,
    inner_max_its: int = 25,
    ksps: Optional[List[PETSc.KSP]] = None,
    ksp_options: Optional[List[List[List[str]]]] = None,
    rhs_tensors: Optional[List[fenics.PETScMatrix]] = None,
    lhs_tensors: Optional[List[fenics.PETScVector]] = None,
    inner_is_linear: bool = False,
) -> None:
    """Solves a system of coupled PDEs via a Picard iteration.

    Args:
        F_list: List of the coupled PDEs.
        u_list: List of the state variables (to be solved for).
        bcs_list: List of boundary conditions for the PDEs.
        max_iter: The maximum number of iterations for the Picard iteration.
        rtol: The relative tolerance for the Picard iteration, default is 1e-10.
        atol: The absolute tolerance for the Picard iteration, default is 1e-10.
        verbose: Boolean flag, if ``True``, output is written to stdout, default is
            ``True``.
        inner_damped: Boolean flag, if ``True``, the inner problems are solved with a
            damped Newton method, default is ``True``
        inner_inexact: Boolean flag, if ``True``, the inner problems are solved with a
            inexact Newton method, default is ``True``
        inner_verbose: Boolean flag, if ``True``, the inner problems write the history
            to stdout, default is ``False``.
        inner_max_its: Maximum number of iterations for the inner Newton solver; default
            is 25.
        ksps: List of PETSc KSP objects for solving the inner (linearized) problems,
            optional. Default is ``None``, in which case the direct solver mumps is
            used.
        ksp_options: List of options for the KSP objects.
        rhs_tensors: List of matrices for the right-hand sides of the inner (linearized)
            equations.
        lhs_tensors: List of vectors for the left-hand sides of the inner (linearized)
            equations.
        inner_is_linear: Boolean flag, if this is ``True``, all problems are actually
            linear ones, and only a linear solver is used.
    """

    F_list = utils.enlist(F_list)
    u_list = utils.enlist(u_list)
    bcs_list = utils._check_and_enlist_bcs(bcs_list)

    if not len(F_list) == len(u_list):
        raise _exceptions.InputError(
            "cashocs.picard_iteration",
            "F_list",
            "Length of F_list and u_list does not match.",
        )

    if not len(bcs_list) == len(u_list):
        raise _exceptions.InputError(
            "cashocs.picard_iteration",
            "bcs_list",
            "Length of bcs_list and u_list does not match.",
        )

    if ksps is None:
        ksps = [None] * len(u_list)
    if ksp_options is None:
        ksp_options = [None] * len(u_list)
    if rhs_tensors is None:
        rhs_tensors = [None] * len(u_list)
    if lhs_tensors is None:
        lhs_tensors = [None] * len(u_list)

    res_tensor = [fenics.PETScVector() for _ in range(len(u_list))]
    eta_max = 0.9
    gamma = 0.9
    res_0 = 1.0
    tol = 1.0

    for i in range(max_iter + 1):
        res = 0.0
        for j in range(len(u_list)):
            fenics.assemble(F_list[j], tensor=res_tensor[j])
            [bc.apply(res_tensor[j]) for bc in bcs_list[j]]

            # TODO: Include very first solve to adjust absolute tolerance
            res += pow(res_tensor[j].norm("l2"), 2)

        if res == 0:
            break

        res = np.sqrt(res)
        if i == 0:
            res_0 = res
            tol = atol + rtol * res_0
        if verbose:
            print(
                f"Picard iteration {i:d}: "
                f"||res|| (abs): {res:.3e}   "
                f"||res|| (rel): {res/res_0:.3e}"
            )
        if res <= tol:
            break

        if i == max_iter:
            raise _exceptions.NotConvergedError("Picard iteration")

        for j in range(len(u_list)):
            eta = np.minimum(gamma * res, eta_max)
            eta = np.minimum(
                eta_max,
                np.maximum(eta, 0.5 * tol / res),
            )

            newton_solver.newton_solve(
                F_list[j],
                u_list[j],
                bcs_list[j],
                rtol=eta,
                atol=atol * 1e-1,
                max_iter=inner_max_its,
                damped=inner_damped,
                inexact=inner_inexact,
                verbose=inner_verbose,
                ksp=ksps[j],
                ksp_options=ksp_options[j],
                rhs_tensor=rhs_tensors[j],
                lhs_tensor=lhs_tensors[j],
                is_linear=inner_is_linear,
            )

    if verbose:
        print("")
