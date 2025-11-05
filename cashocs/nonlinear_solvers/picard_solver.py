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

"""A Picard iteration for coupled PDEs."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs import log
from cashocs.nonlinear_solvers import snes

if TYPE_CHECKING:
    from cashocs import _typing

T = TypeVar("T")


def _setup_obj(obj: T, dim: int) -> T | list[None]:
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
    bcs_list: list[list[fenics.DirichletBC]],
) -> list[list[fenics.DirichletBC]]:
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


def _enlist_picard(obj: list[T | None] | None, length: int) -> list[T | None]:
    if obj is None:
        return [None] * length
    else:
        return obj


def picard_iteration(
    form_list: list[ufl.form] | ufl.Form,
    u_list: list[fenics.Function] | fenics.Function,
    bcs_list: list[fenics.DirichletBC] | list[list[fenics.DirichletBC]],
    max_iter: int = 50,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    verbose: bool = True,
    inner_max_iter: int = 25,
    ksp_options: list[_typing.KspOption] | None = None,
    # pylint: disable=invalid-name
    A_tensors: list[fenics.PETScMatrix] | None = None,
    b_tensors: list[fenics.PETScVector] | None = None,
    preconditioner_forms: list[ufl.Form] | ufl.Form | None = None,
    newton_linearizations: list[ufl.Form] | None = None,
) -> None:
    """Solves a system of coupled PDEs via a Picard iteration.

    Args:
        form_list: list of the coupled PDEs.
        u_list: list of the state variables (to be solved for).
        bcs_list: list of boundary conditions for the PDEs.
        max_iter: The maximum number of iterations for the Picard iteration.
        rtol: The relative tolerance for the Picard iteration, default is 1e-10.
        atol: The absolute tolerance for the Picard iteration, default is 1e-10.
        verbose: Boolean flag, if ``True``, output is written to stdout, default is
            ``True``.
        inner_max_iter: Maximum number of iterations for the inner Newton solver;
            default is 25.
        ksp_options: list of options for the KSP objects.
        A_tensors: list of matrices for the right-hand sides of the inner (linearized)
            equations.
        b_tensors: list of vectors for the left-hand sides of the inner (linearized)
            equations.
        preconditioner_forms: The list of forms for the preconditioner. The default
            is `None`, so that the preconditioner matrix is the same as the system
            matrix.
        newton_linearizations: A list of UFL forms describing which (alternative)
            linearizations should be used for the (nonlinear) equations when
            solving them (with Newton's method). The default is `None`, so that the
            Jacobian of the supplied state forms is used.

    """
    form_list = _utils.enlist(form_list)
    u_list = _utils.enlist(u_list)
    bcs_list = _utils.check_and_enlist_bcs(bcs_list)
    bcs_list_hom = _create_homogenized_bcs(bcs_list)

    preconditioner_form_list = _enlist_picard(preconditioner_forms, len(u_list))
    newton_linearization_list = _enlist_picard(newton_linearizations, len(u_list))

    comm = u_list[0].function_space().mesh().mpi_comm()

    prefix = "Picard iteration:  "

    res_tensor = [fenics.PETScVector(comm) for _ in u_list]
    eta_max = 0.9
    gamma = 0.9
    res_0 = 1.0
    tol = 1.0

    for i in range(max_iter + 1):
        res = _compute_residual(form_list, res_tensor, bcs_list_hom)
        if i == 0:
            res_0 = res
            tol = atol + rtol * res_0

        if i % 10 == 0:
            info_str = f"\n{prefix}iter,  abs. residual,  rel. residual\n\n"
        else:
            info_str = ""
        val_str = f"{prefix}{i:4d},  {res:>13.3e},  {res / res_0:>13.3e}"
        if verbose:
            if comm.rank == 0:
                print(info_str + val_str, flush=True)
            comm.barrier()
        else:
            log.debug(info_str + val_str)

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

            snes.snes_solve(
                form_list[j],
                u_list[j],
                bcs_list[j],
                derivative=newton_linearization_list[j],
                petsc_options=ksp_option,
                rtol=eta,
                atol=atol * 1e-1,
                max_iter=inner_max_iter,
                A_tensor=A_tensor,
                b_tensor=b_tensor,
                preconditioner_form=preconditioner_form_list[j],
            )


def _compute_residual(
    form_list: list[ufl.Form],
    res_tensor: list[fenics.PETScVector],
    bcs_list: list[list[fenics.DirichletBC]],
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
    ksp_options: list[_typing.KspOption] | None,
    # pylint: disable=invalid-name
    A_tensors: list[fenics.PETScMatrix] | None,
    b_tensors: list[fenics.PETScVector] | None,
) -> tuple[_typing.KspOption | None, fenics.PETScMatrix, fenics.PETScVector]:
    """Computes the arguments for the individual components considered in the iteration.

    Returns:
        A tuple [ksp_option, A_tensor, b_tensor].

    """
    ksp_option = ksp_options[j] if ksp_options is not None else None
    # pylint: disable=invalid-name
    A_tensor = A_tensors[j] if A_tensors is not None else None
    b_tensor = b_tensors[j] if b_tensors is not None else None

    return ksp_option, A_tensor, b_tensor
