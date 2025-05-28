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

"""Linear solver for linear PDEs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import _typing


def linear_solve(
    linear_form: ufl.Form,
    u: fenics.Function,
    bcs: fenics.DirichletBC | list[fenics.DirichletBC],
    ksp_options: _typing.KspOption | None = None,
    preconditioner_form: ufl.Form = None,
    A_tensor: fenics.PETScMatrix | None = None,  # pylint: disable=invalid-name
    b_tensor: fenics.PETScVector | None = None,
    linear_solver: _utils.linalg.LinearSolver | None = None,
) -> fenics.Function:
    """Solves a linear problem.

    Args:
        linear_form: The linear variational form of the problem, i.e., linear_form == 0
        u: The function to be solved for
        bcs: The boundary conditions for the problem
        ksp_options: The options for the PETSc KSP solver, optional. Default is `None`,
            where the linear solver MUMPS is used
        preconditioner_form: The UFL form for defining the preconditioner. Must be a
            bilinear form.
        A_tensor: A fenics.PETScMatrix for storing the left-hand side of the linear
            sub-problem.
        b_tensor: A fenics.PETScVector for storing the right-hand side of the linear
            sub-problem.
        linear_solver: The linear solver used to solve the (discretized) linear problem.

    Returns:
        The computed solution, this overwrites the input function `u`.

    """
    lhs_form = _utils.bilinear_boundary_form_modification(
        [fenics.derivative(linear_form, u)]
    )[0]
    rhs_form = -ufl.replace(linear_form, {u: fenics.Constant(np.zeros(u.ufl_shape))})

    assembler = fenics.SystemAssembler(lhs_form, rhs_form, bcs)
    assembler.keep_diagonal = True

    comm = u.function_space().mesh().mpi_comm()
    A_fenics = A_tensor or fenics.PETScMatrix(comm)  # pylint: disable=invalid-name
    b_fenics = b_tensor or fenics.PETScVector(comm)

    assembler.assemble(A_fenics)
    A_fenics.ident_zeros()
    A_matrix = fenics.as_backend_type(A_fenics).mat()  # pylint: disable=invalid-name

    assembler.assemble(b_fenics)
    b = fenics.as_backend_type(b_fenics).vec()

    if preconditioner_form is not None:
        if len(preconditioner_form.arguments()) == 1:
            preconditioner_form = fenics.derivative(preconditioner_form, u)

        P_fenics = fenics.PETScMatrix(comm)  # pylint: disable=invalid-name
        assembler_p = fenics.SystemAssembler(preconditioner_form, rhs_form, bcs)
        assembler_p.keep_diagonal = True

        assembler_p.assemble(P_fenics)
        P_matrix = fenics.as_backend_type(  # pylint: disable=invalid-name
            P_fenics
        ).mat()
    else:
        P_matrix = None  # pylint: disable=invalid-name

    if linear_solver is None:
        linear_solver = _utils.linalg.LinearSolver()
    linear_solver.solve(u, A=A_matrix, b=b, ksp_options=ksp_options, P=P_matrix)

    return u
