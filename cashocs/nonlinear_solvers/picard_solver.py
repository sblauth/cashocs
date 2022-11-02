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

"""A Picard iteration for coupled PDEs."""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, TypeVar, Union

import fenics
import numpy as np
import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs.nonlinear_solvers import newton_solver

if TYPE_CHECKING:
    from cashocs import _typing

T = TypeVar("T")


def _setup_obj(obj: T, dim: int) -> Union[T, List[None]]:
    """Returns a list of None if obj is None, else returns obj.

    Args:
        obj: The object which is checked.
        dim: The dimension of the list.

    Returns:
        Either the obj (if not None) or a list of None.

    """
    if obj is None:
        return [None] * dim
    else:
        return obj


def _create_homogenized_bcs(
    bcs_list: List[List[fenics.DirichletBC]],
) -> List[List[fenics.DirichletBC]]:
    """Copies the bcs_list and homogenizes the boundary conditions.

    Args:
        bcs_list: The list of boundary conditions

    Returns:
        The homogenized list of boundary conditions

    """
    bcs_list_hom = []
    for i in range(len(bcs_list)):
        temp_list = []
        for bc in bcs_list[i]:
            bc_hom = fenics.DirichletBC(bc)
            bc_hom.homogenize()
            temp_list.append(bc_hom)
        bcs_list_hom.append(temp_list)

    return bcs_list_hom


def picard_iteration(
    form_list: Union[List[ufl.form], ufl.Form],
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
    ksp_options: Optional[_typing.KspOptions] = None,
    # pylint: disable=invalid-name
    A_tensors: Optional[List[fenics.PETScMatrix]] = None,
    b_tensors: Optional[List[fenics.PETScVector]] = None,
    inner_is_linear: bool = False,
) -> None:
    """Solves a system of coupled PDEs via a Picard iteration.

    Args:
        form_list: List of the coupled PDEs.
        u_list: List of the state variables (to be solved for).
        bcs_list: List of boundary conditions for the PDEs.
        max_iter: The maximum number of iterations for the Picard iteration.
        rtol: The relative tolerance for the Picard iteration, default is 1e-10.
        atol: The absolute tolerance for the Picard iteration, default is 1e-10.
        verbose: Boolean flag, if ``True``, output is written to stdout, default is
            ``True``.
        inner_damped: Boolean flag, if ``True``, the inner problems are solved with a
            damped Newton method, default is ``True``
        inner_inexact: Boolean flag, if ``True``, the inner problems are solved with an
            inexact Newton method, default is ``True``
        inner_verbose: Boolean flag, if ``True``, the inner problems write the history
            to stdout, default is ``False``.
        inner_max_its: Maximum number of iterations for the inner Newton solver; default
            is 25.
        ksp_options: List of options for the KSP objects.
        A_tensors: List of matrices for the right-hand sides of the inner (linearized)
            equations.
        b_tensors: List of vectors for the left-hand sides of the inner (linearized)
            equations.
        inner_is_linear: Boolean flag, if this is ``True``, all problems are actually
            linear ones, and only a linear solver is used.

    """
    is_printing = verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0
    form_list = _utils.enlist(form_list)
    u_list = _utils.enlist(u_list)
    bcs_list = _utils.check_and_enlist_bcs(bcs_list)
    bcs_list_hom = _create_homogenized_bcs(bcs_list)

    prefix = "Picard iteration:  "

    res_tensor = [fenics.PETScVector() for _ in range(len(u_list))]
    eta_max = 0.9
    gamma = 0.9
    res_0 = 1.0
    tol = 1.0

    for i in range(max_iter + 1):
        res = _compute_residual(form_list, res_tensor, bcs_list_hom)
        if i == 0:
            res_0 = res
            tol = atol + rtol * res_0
        if is_printing:
            if i % 10 == 0:
                info_str = f"\n{prefix}iter,  abs. residual,  rel. residual\n\n"
            else:
                info_str = ""
            val_str = f"{prefix}{i:4d},  {res:>13.3e},  {res/res_0:>13.3e}"

            print(info_str + val_str, flush=True)
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

            ksp_option, A_tensor, b_tensor = _get_linear_solver_options(
                j, ksp_options, A_tensors, b_tensors
            )

            newton_solver.newton_solve(
                form_list[j],
                u_list[j],
                bcs_list[j],
                rtol=eta,
                atol=atol * 1e-1,
                max_iter=inner_max_its,
                damped=inner_damped,
                inexact=inner_inexact,
                verbose=inner_verbose,
                ksp_options=ksp_option,
                A_tensor=A_tensor,
                b_tensor=b_tensor,
                is_linear=inner_is_linear,
            )

    if is_printing:
        print("", flush=True)


def _compute_residual(
    form_list: List[ufl.Form],
    res_tensor: List[fenics.PETScVector],
    bcs_list: List[List[fenics.DirichletBC]],
) -> float:
    """Computes the residual for the picard iteration.

    Args:
        form_list: The list of forms which make the system.
        res_tensor: The vectors into which the residual shall be assembled.
        bcs_list: The list of boundary conditions for the system.

    Returns:
        The residual of the system to be solved with a Picard iteration.

    """
    res = 0.0
    for j in range(len(form_list)):
        fenics.assemble(form_list[j], tensor=res_tensor[j])
        for bc in bcs_list[j]:
            bc.apply(res_tensor[j])

        # TODO: Include very first solve to adjust absolute tolerance
        res += pow(res_tensor[j].norm("l2"), 2)

    result: float = np.sqrt(res)
    return result


def _get_linear_solver_options(
    j: int,
    ksp_options: Optional[_typing.KspOptions],
    # pylint: disable=invalid-name
    A_tensors: Optional[List[fenics.PETScMatrix]],
    b_tensors: Optional[List[fenics.PETScVector]],
) -> Tuple[
    Optional[List[List[Union[str, int, float]]]], fenics.PETScMatrix, fenics.PETScVector
]:
    """Computes the arguments for the individual components considered in the iteration.

    Returns:
        A tuple [ksp_option, A_tensor, b_tensor].

    """
    ksp_option = ksp_options[j] if ksp_options is not None else None
    # pylint: disable=invalid-name
    A_tensor = A_tensors[j] if A_tensors is not None else None
    b_tensor = b_tensors[j] if b_tensors is not None else None

    return ksp_option, A_tensor, b_tensor
