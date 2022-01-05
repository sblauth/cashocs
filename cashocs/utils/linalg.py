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


from __future__ import annotations

from typing import Union, List, Tuple, Optional

import fenics
import ufl
from petsc4py import PETSc

from .._exceptions import InputError, PETScKSPError


def _assemble_petsc_system(
    A_form: ufl.Form,
    b_form: ufl.Form,
    bcs: Optional[Union[fenics.DirichletBC, List[fenics.DirichletBC]]] = None,
    A_tensor: Optional[fenics.PETScMatrix] = None,
    b_tensor: Optional[fenics.PETScVector] = None,
) -> Tuple[PETSc.Mat, PETSc.Vec]:
    """Assembles a system symmetrically and converts objects to PETSc format.

    Parameters
    ----------
    A_form : ufl.form.Form
        The UFL form for the left-hand side of the linear equation.
    b_form : ufl.form.Form
        The UFL form for the right-hand side of the linear equation.
    bcs : None or fenics.DirichletBC or list[fenics.DirichletBC], optional
        A list of Dirichlet boundary conditions.
    A_tensor : fenics.PETScMatrix or None, optional
        A matrix into which the result is assembled. Default is ``None``
    b_tensor : fenics.PETScVector or None, optional
        A vector into which the result is assembled. Default is ``None``

    Returns
    -------
    petsc4py.PETSc.Mat
        The petsc matrix for the left-hand side of the linear equation.
    petsc4py.PETSc.Vec
        The petsc vector for the right-hand side of the linear equation.

    Notes
    -----
    This function always uses the ident_zeros method of the matrix in order to add a one to the diagonal
    in case the corresponding row only consists of zeros. This allows for well-posed problems on the
    boundary etc.
    """

    if A_tensor is None:
        A_tensor = fenics.PETScMatrix()
    if b_tensor is None:
        b_tensor = fenics.PETScVector()
    fenics.assemble_system(
        A_form, b_form, bcs, keep_diagonal=True, A_tensor=A_tensor, b_tensor=b_tensor
    )
    A_tensor.ident_zeros()

    A = A_tensor.mat()
    b = b_tensor.vec()

    return A, b


def _setup_petsc_options(
    ksps: List[PETSc.KSP], ksp_options: List[List[List[str]]]
) -> None:
    """Sets up an (iterative) linear solver.

    This is used to pass user defined command line type options for PETSc
    to the PETSc KSP objects. Here, options[i] is applied to ksps[i]

    Parameters
    ----------
    ksps : list[petsc4py.PETSc.KSP]
        A list of PETSc KSP objects (linear solvers) to which the (command line)
        options are applied to.
    ksp_options : list[list[list[str]]]
        A list of command line options that specify the iterative solver
        from PETSc.

    Returns
    -------
    None
    """

    opts = fenics.PETScOptions

    for i in range(len(ksps)):
        opts.clear()

        for option in ksp_options[i]:
            opts.set(*option)

        ksps[i].setFromOptions()


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

    Parameters
    ----------
    ksp : petsc4py.PETSc.KSP or None, optional
        The PETSc KSP object used to solve the problem. None means that the solver
        mumps is used (default is None).
    A : petsc4py.PETSc.Mat or None, optional
        The PETSc matrix corresponding to the left-hand side of the problem. If
        this is None, then the matrix stored in the ksp object is used. Raises
        an error if no matrix is stored. Default is None.
    b : petsc4py.PETSc.Vec or None, optional
        The PETSc vector corresponding to the right-hand side of the problem.
        If this is None, then a zero right-hand side is assumed, and a zero
        vector is returned. Default is None.
    x : petsc4py.PETSc.Vec or None, optional
        The PETSc vector that stores the solution of the problem. If this is
        None, then a new vector will be created (and returned)
    ksp_options : list or None, optional
        The options for the PETSc ksp object. If this is None (the default) a direct
        method is used
    rtol : float or None, optional
        The relative tolerance used in case an iterative solver is used for solving the
        linear problem. Overrides the specification in the ksp object and ksp_options.
    atol : float or None, optional
        The absolute tolerance used in case an iterative solver is used for solving the
        linear problem. Overrides the specification in the ksp object and ksp_options.

    Returns
    -------
    petsc4py.PETSc.Vec
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
            raise InputError(
                "cashocs.utils._solve_linear_problem",
                "ksp",
                "The KSP object has to be initialized with some Matrix in case A is None.",
            )

    if b is None:
        return A.getVecs()[0]

    if x is None:
        x, _ = A.getVecs()

    if ksp_options is not None:
        opts = fenics.PETScOptions
        opts.clear()

        for option in ksp_options:
            opts.set(*option)

        ksp.setFromOptions()

    if rtol is not None:
        ksp.rtol = rtol
    if atol is not None:
        ksp.atol = atol
    ksp.solve(b, x)

    if ksp.getConvergedReason() < 0:
        raise PETScKSPError(ksp.getConvergedReason())

    return x


class Interpolator:
    """Efficient interpolation between two function spaces.

    This is very useful, if multiple interpolations have to be
    carried out between the same spaces, which is made significantly
    faster by computing the corresponding matrix.
    The function spaces can even be defined on different meshes.

    Notes
    -----

    This class only works properly for continuous Lagrange elements and
    constant, discontinuous Lagrange elements.

    Examples
    --------
    Here, we consider interpolating from CG1 elements to CG2 elements ::

        from fenics import *
        import cashocs

        mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
        V1 = FunctionSpace(mesh, 'CG', 1)
        V2 = FunctionSpace(mesh, 'CG', 2)

        expr = Expression('sin(2*pi*x[0])', degree=1)
        u = interpolate(expr, V1)

        interp = cashocs.utils.Interpolator(V1, V2)
        interp.interpolate(u)
    """

    def __init__(self, V: fenics.FunctionSpace, W: fenics.FunctionSpace) -> None:
        """
        Parameters
        ----------
        V : fenics.FunctionSpace
            The function space whose objects shall be interpolated.
        W : fenics.FunctionSpace
            The space into which they shall be interpolated.
        """

        if not (
            V.ufl_element().family() == "Lagrange"
            or (
                V.ufl_element().family() == "Discontinuous Lagrange"
                and V.ufl_element().degree() == 0
            )
        ):
            raise InputError(
                "cashocs.utils.Interpolator",
                "V",
                "The interpolator only works with CG n or DG 0 elements",
            )
        if not (
            W.ufl_element().family() == "Lagrange"
            or (
                W.ufl_element().family() == "Discontinuous Lagrange"
                and W.ufl_element().degree() == 0
            )
        ):
            raise InputError(
                "cashocs.utils.Interpolator",
                "W",
                "The interpolator only works with CG n or DG 0 elements",
            )

        self.V = V
        self.W = W
        self.transfer_matrix = fenics.PETScDMCollection.create_transfer_matrix(
            self.V, self.W
        )

    def interpolate(self, u: fenics.Function) -> fenics.Function:
        """Interpolates function to target space.

        The function has to belong to the origin space, i.e., the first argument
        of __init__, and it is interpolated to the destination space, i.e., the
        second argument of __init__. There is no need to call set_allow_extrapolation
        on the function (this is done automatically due to the method).

        Parameters
        ----------
        u : fenics.Function
            The function that shall be interpolated.

        Returns
        -------
        fenics.Function
            The result of the interpolation.
        """

        if not u.function_space() == self.V:
            raise InputError(
                "cashocs.utils.Interpolator.interpolate",
                "u",
                "The input does not belong to the correct function space.",
            )
        v = fenics.Function(self.W)
        v.vector()[:] = (self.transfer_matrix * u.vector())[:]

        return v
