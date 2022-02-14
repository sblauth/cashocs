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

"""Module for performing linear algebra tasks."""

from __future__ import annotations

from typing import Union, List, Tuple, Optional

import fenics
import numpy as np
import ufl
from petsc4py import PETSc

from cashocs import _exceptions


def _split_linear_forms(forms: List[ufl.Form]) -> Tuple[List[ufl.Form], List[ufl.Form]]:
    """Splits a list of linear forms into left- and right-hand sides.

    Args:
        forms: A list of (linear) ufl forms.

    Returns:
        A tuple (lhs_forms, rhs_forms), where lhs_forms is the list of forms of the
        left-hand sides, and rhs_forms is the list of forms of the right-hand side.
    """
    lhs_list = []
    rhs_list = []
    for i in range(len(forms)):
        try:
            lhs, rhs = fenics.system(forms[i])
        except ufl.log.UFLException:
            raise _exceptions.CashocsException(
                "The state system could not be transferred to a linear "
                "system.\n"
                "Perhaps you specified that the system is linear, "
                "although it is not.\n"
                "In your config, in the StateSystem section, "
                "try using is_linear = False."
            )
        lhs_list.append(lhs)

        if rhs.empty():
            test_function = lhs.arguments()[0]
            mesh = lhs.ufl_domain()
            dx = fenics.Measure("dx", mesh)
            zero_form = (
                fenics.dot(
                    fenics.Constant(np.zeros(test_function.ufl_shape)),
                    test_function,
                )
                * dx
            )
            rhs_list.append(zero_form)
        else:
            rhs_list.append(rhs)

    return lhs_list, rhs_list


# noinspection PyUnresolvedReferences,PyPep8Naming
def _assemble_petsc_system(
    lhs_form: ufl.Form,
    rhs_form: ufl.Form,
    bcs: Optional[Union[fenics.DirichletBC, List[fenics.DirichletBC]]] = None,
    A_tensor: Optional[fenics.PETScMatrix] = None,
    b_tensor: Optional[fenics.PETScVector] = None,
) -> Tuple[PETSc.Mat, PETSc.Vec]:
    """Assembles a system symmetrically and converts objects to PETSc format.

    Args:
        lhs_form: The UFL form for the left-hand side of the linear equation.
        rhs_form: The UFL form for the right-hand side of the linear equation.
        bcs: A list of Dirichlet boundary conditions.
        A_tensor: A matrix into which the result is assembled. Default is ``None``.
        b_tensor: A vector into which the result is assembled. Default is ``None``.

    Returns:
        A tuple (A, b), where A is the matrix of the linear system, and b is the vector
        of the linear system.

    Notes:
        This function always uses the ident_zeros method of the matrix in order to add a
        one to the diagonal in case the corresponding row only consists of zeros. This
        allows for well-posed problems on the boundary etc.
    """
    if A_tensor is None:
        # noinspection PyPep8Naming
        A_tensor = fenics.PETScMatrix()
    if b_tensor is None:
        b_tensor = fenics.PETScVector()
    fenics.assemble_system(
        lhs_form,
        rhs_form,
        bcs,
        keep_diagonal=True,
        A_tensor=A_tensor,
        b_tensor=b_tensor,
    )
    A_tensor.ident_zeros()

    # noinspection PyPep8Naming
    A = A_tensor.mat()
    b = b_tensor.vec()

    return A, b


# noinspection PyUnresolvedReferences
def _setup_petsc_options(
    ksps: List[PETSc.KSP], ksp_options: List[List[List[str]]]
) -> None:
    """Sets up an (iterative) linear solver.

    This is used to pass user defined command line type options for PETSc
    to the PETSc KSP objects. Here, options[i] is applied to ksps[i].

    Args:
        ksps: A list of PETSc KSP objects (linear solvers) to which the (command line)
            options are applied to.
        ksp_options: A list of command line options that specify the iterative solver
            from PETSc.
    """
    opts = fenics.PETScOptions

    for i in range(len(ksps)):
        # noinspection PyArgumentList
        opts.clear()

        for option in ksp_options[i]:
            opts.set(*option)

        ksps[i].setFromOptions()


# noinspection PyPep8Naming,PyUnresolvedReferences
def _solve_linear_problem(
    ksp: Optional[PETSc.KSP] = None,
    A: Optional[PETSc.Mat] = None,
    b: Optional[PETSc.Vec] = None,
    x: Optional[PETSc.Vec] = None,
    ksp_options: Optional[List[List[str]]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> PETSc.Vec:
    """Solves a finite dimensional linear problem.

    Args:
        ksp: The PETSc KSP object used to solve the problem. None means that the solver
            mumps is used (default is None).
        A: The PETSc matrix corresponding to the left-hand side of the problem. If
            this is None, then the matrix stored in the ksp object is used. Raises
            an error if no matrix is stored. Default is None.
        b: The PETSc vector corresponding to the right-hand side of the problem.
            If this is None, then a zero right-hand side is assumed, and a zero vector
            is returned. Default is None.
        x: The PETSc vector that stores the solution of the problem. If this is
            None, then a new vector will be created (and returned).
        ksp_options: The options for the PETSc ksp object. If this is None (the default)
            a direct method is used.
        rtol: The relative tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.
        atol: The absolute tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.

    Returns:
        The solution vector.
    """
    if ksp is None:
        ksp = PETSc.KSP().create()
        options = [
            ["ksp_type", "preonly"],
            ["pc_type", "lu"],
            ["pc_factor_mat_solver_type", "mumps"],
            ["mat_mumps_icntl_24", 1],
        ]

        _setup_petsc_options([ksp], [options])

    if A is not None:
        ksp.setOperators(A)
    else:
        A = ksp.getOperators()[0]
        if A.size[0] == -1 and A.size[1] == -1:
            raise _exceptions.InputError(
                "cashocs.utils._solve_linear_problem",
                "ksp",
                "The KSP object has to be initialized with some Matrix in case A is "
                "None.",
            )

    if b is None:
        return A.getVecs()[0]

    if x is None:
        x, _ = A.getVecs()

    if ksp_options is not None:
        _setup_petsc_options([ksp], [ksp_options])

    if rtol is not None:
        ksp.rtol = rtol
    if atol is not None:
        ksp.atol = atol
    ksp.solve(b, x)

    if ksp.getConvergedReason() < 0:
        raise _exceptions.PETScKSPError(ksp.getConvergedReason())

    return x


# noinspection PyPep8Naming,PyUnresolvedReferences
def _assemble_and_solve_linear(
    lhs_form: ufl.Form,
    rhs_form: ufl.Form,
    bcs: Optional[Union[fenics.DirichletBC, List[fenics.DirichletBC]]] = None,
    A: Optional[fenics.PETScMatrix] = None,
    b: Optional[fenics.PETScVector] = None,
    x: Optional[PETSc.Vec] = None,
    ksp: Optional[PETSc.KSP] = None,
    ksp_options: Optional[List[List[str]]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> PETSc.Vec:
    """Assembles and solves a linear system.

    Args:
        lhs_form: The UFL form for the left-hand side of the linear equation.
        rhs_form: The UFL form for the right-hand side of the linear equation.
        bcs: A list of Dirichlet boundary conditions.
        A: A matrix into which the lhs is assembled. Default is ``None``.
        b: A vector into which the rhs is assembled. Default is ``None``.
        x: The PETSc vector that stores the solution of the problem. If this is
            None, then a new vector will be created (and returned).
        ksp: The PETSc KSP object used to solve the problem. None means that the solver
            mumps is used (default is None).
        ksp_options: The options for the PETSc ksp object. If this is None (the default)
            a direct method is used.
        rtol: The relative tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.
        atol: The absolute tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.

    Returns:
        A PETSc vector containing the solution x.
    """
    # noinspection PyPep8Naming
    A_matrix, b_vector = _assemble_petsc_system(
        lhs_form, rhs_form, bcs, A_tensor=A, b_tensor=b
    )
    solution = _solve_linear_problem(
        ksp=ksp,
        A=A_matrix,
        b=b_vector,
        x=x,
        ksp_options=ksp_options,
        rtol=rtol,
        atol=atol,
    )

    return solution


class Interpolator:
    """Efficient interpolation between two function spaces.

    This is very useful, if multiple interpolations have to be carried out between the
    same spaces, which is made significantly faster by computing the corresponding
    matrix. The function spaces can even be defined on different meshes.

    Notes:
        This class only works properly for continuous Lagrange elements and constant,
        discontinuous Lagrange elements.

    Examples:
        Here, we consider interpolating from CG1 elements to CG2 elements ::

            import fenics
            import cashocs

            mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
            V1 = fenics.FunctionSpace(mesh, 'CG', 1)
            V2 = fenics.FunctionSpace(mesh, 'CG', 2)

            expr = fenics.Expression('sin(2*pi*x[0])', degree=1)
            u = fenics.interpolate(expr, V1)

            interp = cashocs.utils.Interpolator(V1, V2)
            interp.interpolate(u)
    """

    def __init__(
        self, origin_space: fenics.FunctionSpace, target_space: fenics.FunctionSpace
    ) -> None:
        """Initializes self.

        Args:
            origin_space: The function space whose objects shall be interpolated.
            target_space: The space into which they shall be interpolated.
        """
        if not (
            origin_space.ufl_element().family() == "Lagrange"
            or (
                origin_space.ufl_element().family() == "Discontinuous Lagrange"
                and origin_space.ufl_element().degree() == 0
            )
        ):
            raise _exceptions.InputError(
                "cashocs.utils.Interpolator",
                "origin_space",
                "The interpolator only works with CG n or DG 0 elements",
            )
        if not (
            target_space.ufl_element().family() == "Lagrange"
            or (
                target_space.ufl_element().family() == "Discontinuous Lagrange"
                and target_space.ufl_element().degree() == 0
            )
        ):
            raise _exceptions.InputError(
                "cashocs.utils.Interpolator",
                "target_space",
                "The interpolator only works with CG n or DG 0 elements",
            )

        self.origin_space = origin_space
        self.target_space = target_space
        # noinspection PyTypeChecker
        self.transfer_matrix = fenics.PETScDMCollection.create_transfer_matrix(
            self.origin_space, self.target_space
        )

    def interpolate(self, u: fenics.Function) -> fenics.Function:
        """Interpolates function to target space.

        The function has to belong to the origin space, i.e., the first argument
        of __init__, and it is interpolated to the destination space, i.e., the
        second argument of __init__. There is no need to call set_allow_extrapolation
        on the function (this is done automatically due to the method).

        Args:
            u: The function that shall be interpolated.

        Returns:
            The result of the interpolation.
        """
        v = fenics.Function(self.target_space)
        v.vector()[:] = (self.transfer_matrix * u.vector())[:]

        return v
