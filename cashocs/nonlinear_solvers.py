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

"""Custom solvers for nonlinear equations.

This module has custom solvers for nonlinear PDEs, including a damped
Newton methd. This is the only function at the moment, others might
follow.
"""

from __future__ import annotations

from typing import List, Union, Optional

import fenics
import numpy as np
import ufl
from petsc4py import PETSc
from typing_extensions import Literal

from ._exceptions import InputError, NotConvergedError
from ._loggers import warning
from .utils import _setup_petsc_options, _solve_linear_problem


def newton_solve(
    F: ufl.Form,
    u: fenics.Function,
    bcs: Union[fenics.DirichletBC, List[fenics.DirichletBC]],
    dF: Optional[ufl.Form] = None,
    shift: Optional[ufl.Form] = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_iter: int = 50,
    convergence_type: Literal["combined", "rel", "abs"] = "combined",
    norm_type: Literal["l2", "linf"] = "l2",
    damped: bool = True,
    inexact: bool = True,
    verbose: bool = True,
    ksp: Optional[PETSc.KSP] = None,
    ksp_options: Optional[List[List[str]]] = None,
    A_tensor: Optional[fenics.PETScMatrix] = None,
    b_tensor: Optional[fenics.PETScVector] = None,
) -> fenics.Function:
    r"""A damped Newton method for solving nonlinear equations.
    
    The damped Newton method is based on the natural monotonicity test from
    `Deuflhard, Newton methods for nonlinear problems <https://doi.org/10.1007/978-3-642-23899-4>`_.
    It also allows fine tuning via a direct interface, and absolute, relative,
    and combined stopping criteria. Can also be used to specify the solver for
    the inner (linear) subproblems via petsc ksps.

    The method terminates after ``max_iter`` iterations, or if a termination criterion is
    satisfied. These criteria are given by

    - a relative one in case ``convergence_type = 'rel'``, i.e.,

    .. math:: \lvert\lvert F_{k} \rvert\rvert \leq \texttt{rtol} \lvert\lvert F_0 \rvert\rvert.

    - an absolute one in case ``convergence_type = 'abs'``, i.e.,

    .. math:: \lvert\lvert F_{k} \rvert\rvert \leq \texttt{atol}.

    - a combination of both in case ``convergence_type = 'combined'``, i.e.,

    .. math:: \lvert\lvert F_{k} \rvert\rvert \leq \texttt{atol} + \texttt{rtol} \lvert\lvert F_0 \rvert\rvert.

    The norm chosen for the termination criterion is specified via ``norm_type``.

    Parameters
    ----------
    F : ufl.form.Form
        The variational form of the nonlinear problem to be solved by Newton's method.
    u : fenics.Function
        The sought solution / initial guess. It is not assumed that the initial guess
        satisfies the Dirichlet boundary conditions, they are applied automatically.
        The method overwrites / updates this Function.
    bcs : list[fenics.DirichletBC]
        A list of DirichletBCs for the nonlinear variational problem.
    dF : ufl.Form or None, optional
        The Jacobian of F, used for the Newton method. Default is None, and in this case
        the Jacobian is computed automatically with AD.
    shift : ufl.Form or None, optional
        This is used in case we want to solve a nonlinear operator equation with a nonlinear
        part ``F`` and a part ``shift``, which does not depend on the variable ``u``.
        Solves the equation :math:`F(u) = shift`. In case shift is ``None`` (the default),
        the equation :math:`F(u) = 0` is solved.
    rtol : float, optional
        Relative tolerance of the solver if convergence_type is either ``'combined'`` or ``'rel'``
        (default is ``rtol = 1e-10``).
    atol : float, optional
        Absolute tolerance of the solver if convergence_type is either ``'combined'`` or ``'abs'``
        (default is ``atol = 1e-10``).
    max_iter : int, optional
        Maximum number of iterations carried out by the method
        (default is ``max_iter = 50``).
    convergence_type : {'combined', 'rel', 'abs'}
        Determines the type of stopping criterion that is used.
    norm_type : {'l2', 'linf'}
        Determines which norm is used in the stopping criterion.
    damped : bool, optional
        If ``True``, then a damping strategy is used. If ``False``, the classical
        Newton-Raphson iteration (without damping) is used (default is ``True``).
    inexact : bool, optional
        If ``True``, then an inexact Newtons method is used, in case an iterative solver
        is used for the inner solution of the linear systems. Default is ``True``.
    verbose : bool, optional
        If ``True``, prints status of the iteration to the console (default
        is ``True``).
    ksp : petsc4py.PETSc.KSP or None, optional
        The PETSc ksp object used to solve the inner (linear) problem
        if this is ``None`` it uses the direct solver MUMPS (default is
        ``None``).
    ksp_options : list[list[str]] or None, optional
        The list of options for the linear solver.


    Returns
    -------
    fenics.Function
        The solution of the nonlinear variational problem, if converged.
        This overrides the input function u.


    Examples
    --------
    Consider the problem

    .. math::
        \begin{alignedat}{2}
        - \Delta u + u^3 &= 1 \quad &&\text{ in } \Omega=(0,1)^2 \\
        u &= 0 \quad &&\text{ on } \Gamma.
        \end{alignedat}
    
    This is solved with the code ::

        from fenics import *
        import cashocs

        mesh, _, boundaries, dx, _, _ = cashocs.regular_mesh(25)
        V = FunctionSpace(mesh, 'CG', 1)

        u = Function(V)
        v = TestFunction(V)
        F = inner(grad(u), grad(v))*dx + pow(u,3)*v*dx - Constant(1)*v*dx
        bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1,2,3,4])
        cashocs.damped_newton_solve(F, u, bcs)
    """

    if isinstance(bcs, fenics.DirichletBC):
        bcs = [bcs]

    if not convergence_type in ["rel", "abs", "combined"]:
        raise InputError(
            "cashocs.nonlinear_solvers.damped_newton_solve",
            "convergence_type",
            "Input convergence_type has to be one of 'rel', 'abs', or 'combined'.",
        )

    if not norm_type in ["l2", "linf"]:
        raise InputError(
            "cashocs.nonlinear_solvers.damped_newton_solve",
            "norm_type",
            "Input norm_type has to be one of 'l2' or 'linf'.",
        )

    # create the PETSc ksp
    if ksp is None:
        if ksp_options is None:
            ksp_options = [
                ["ksp_type", "preonly"],
                ["pc_type", "lu"],
                ["pc_factor_mat_solver_type", "mumps"],
                ["mat_mumps_icntl_24", 1],
            ]

        ksp = PETSc.KSP().create()
        _setup_petsc_options([ksp], [ksp_options])
        ksp.setFromOptions()

    # Calculate the Jacobian.
    if dF is None:
        dF = fenics.derivative(F, u)

    # Setup increment and function for monotonicity test
    V = u.function_space()
    du = fenics.Function(V)
    ddu = fenics.Function(V)
    u_save = fenics.Function(V)

    iterations = 0

    [bc.apply(u.vector()) for bc in bcs]
    # copy the boundary conditions and homogenize them for the increment
    bcs_hom = [fenics.DirichletBC(bc) for bc in bcs]
    [bc.homogenize() for bc in bcs_hom]

    # inexact newton parameters
    eta = 1.0
    eta_max = 0.9999
    eta_a = 1.0
    gamma = 0.9

    assembler = fenics.SystemAssembler(dF, -F, bcs_hom)
    assembler.keep_diagonal = True

    if A_tensor is None:
        A_fenics = fenics.PETScMatrix()
    else:
        A_fenics = A_tensor

    if b_tensor is None:
        residual = fenics.PETScVector()
    else:
        residual = b_tensor

    # Compute the initial residual
    assembler.assemble(residual)

    if shift is not None:
        assembler_shift = fenics.SystemAssembler(dF, shift, bcs_hom)
        residual_shift = fenics.PETScVector()
        assembler_shift.assemble(residual_shift)
        residual[:] += residual_shift[:]

    b = fenics.as_backend_type(residual).vec()

    res_0 = residual.norm(norm_type)
    if res_0 == 0.0:
        if verbose:
            print("Residual vanishes, input is already a solution.")
        return u

    res = res_0
    if verbose:
        print(
            f"Newton Iteration {iterations:2d} - residual (abs):  {res:.3e} (tol = {atol:.3e})    residual (rel):  {res / res_0:.3e} (tol = {rtol:.3e})"
        )

    if convergence_type == "abs":
        tol = atol
    elif convergence_type == "rel":
        tol = rtol * res_0
    else:
        tol = rtol * res_0 + atol

    # While loop until termination
    while res > tol and iterations < max_iter:
        assembler.assemble(A_fenics)
        A_fenics.ident_zeros()
        A = fenics.as_backend_type(A_fenics).mat()

        iterations += 1
        lmbd = 1.0
        breakdown = False
        u_save.vector().vec().aypx(0.0, u.vector().vec())

        # Solve the inner problem
        if inexact:
            if iterations == 1:
                eta = eta_max
            elif gamma * pow(eta, 2) <= 0.1:
                eta = np.minimum(eta_max, eta_a)
            else:
                eta = np.minimum(
                    eta_max,
                    np.maximum(eta_a, gamma * pow(eta, 2)),
                )

            eta = np.minimum(
                eta_max,
                np.maximum(eta, 0.5 * tol / res),
            )
        else:
            eta = rtol * 1e-1

        _solve_linear_problem(ksp, A, b, du.vector().vec(), ksp_options, rtol=eta)
        du.vector().apply("")

        # perform backtracking in case damping is used
        if damped:
            while True:
                u.vector().vec().axpy(lmbd, du.vector().vec())
                assembler.assemble(residual)
                if shift is not None:
                    assembler_shift.assemble(residual_shift)
                    residual[:] += residual_shift[:]
                b = fenics.as_backend_type(residual).vec()
                _solve_linear_problem(
                    ksp=ksp,
                    b=b,
                    x=ddu.vector().vec(),
                    ksp_options=ksp_options,
                    rtol=eta,
                )
                ddu.vector().apply("")

                if ddu.vector().norm(norm_type) / du.vector().norm(norm_type) <= 1:
                    break
                else:
                    u.vector().vec().aypx(0.0, u_save.vector().vec())
                    lmbd /= 2

                if lmbd < 1e-6:
                    breakdown = True
                    break

        else:
            u.vector().vec().axpy(1.0, du.vector().vec())

        if breakdown:
            raise NotConvergedError(
                "Newton solver (state system)", "Stepsize for increment too low."
            )

        if iterations == max_iter:
            raise NotConvergedError(
                "Newton solver (state system)",
                "Maximum number of iterations were exceeded.",
            )

        # compute the new residual
        assembler.assemble(residual)
        if shift is not None:
            assembler_shift.assemble(residual_shift)
            residual[:] += residual_shift[:]
        b = fenics.as_backend_type(residual).vec()

        [bc.apply(residual) for bc in bcs_hom]

        res_prev = res
        res = residual.norm(norm_type)
        eta_a = gamma * pow(res / res_prev, 2)
        if verbose:
            print(
                f"Newton Iteration {iterations:2d} - residual (abs):  {res:.3e} (tol = {atol:.3e})    residual (rel):  {res / res_0:.3e} (tol = {rtol:.3e})"
            )

        if res <= tol:
            if verbose:
                print(f"\nNewton Solver converged after {iterations:d} iterations.\n")
            break

    return u


# deprecated
def damped_newton_solve(
    F: ufl.Form,
    u: fenics.Function,
    bcs: Union[fenics.DirichletBC, List[fenics.DirichletBC]],
    dF: Optional[ufl.Form] = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_iter: int = 50,
    convergence_type: Literal["combined", "rel", "abs"] = "combined",
    norm_type: Literal["l2", "linf"] = "l2",
    damped: bool = True,
    verbose: bool = True,
    ksp: Optional[PETSc.KSP] = None,
    ksp_options: Optional[List[List[str]]] = None,
) -> fenics.Function:
    """Damped Newton solve interface, only here for compatibility reasons.

    Parameters
    ----------
    F : ufl.form.Form
            The variational form of the nonlinear problem to be solved by Newton's method.
    u : fenics.Function
            The sought solution / initial guess. It is not assumed that the initial guess
            satisfies the Dirichlet boundary conditions, they are applied automatically.
            The method overwrites / updates this Function.
    bcs : list[fenics.DirichletBC]
            A list of DirichletBCs for the nonlinear variational problem.
    dF : ufl.form.Form, optional
        The Jacobian of F, used for the Newton method. Default is None, and in this case
        the Jacobian is computed automatically with AD.
    rtol : float, optional
            Relative tolerance of the solver if convergence_type is either ``'combined'`` or ``'rel'``
            (default is ``rtol = 1e-10``).
    atol : float, optional
            Absolute tolerance of the solver if convergence_type is either ``'combined'`` or ``'abs'``
            (default is ``atol = 1e-10``).
    max_iter : int, optional
            Maximum number of iterations carried out by the method
            (default is ``max_iter = 50``).
    convergence_type : {'combined', 'rel', 'abs'}
            Determines the type of stopping criterion that is used.
    norm_type : {'l2', 'linf'}
            Determines which norm is used in the stopping criterion.
    damped : bool, optional
            If ``True``, then a damping strategy is used. If ``False``, the classical
            Newton-Raphson iteration (without damping) is used (default is ``True``).
    verbose : bool, optional
            If ``True``, prints status of the iteration to the console (default
            is ``True``).
    ksp : petsc4py.PETSc.KSP, optional
            The PETSc ksp object used to solve the inner (linear) problem
            if this is ``None`` it uses the direct solver MUMPS (default is
            ``None``).
    ksp_options : list[list[str]]
            The list of options for the linear solver.


    Returns
    -------
    fenics.Function
        The solution of the nonlinear variational problem, if converged.
        This overrides the input function u.


    .. deprecated:: 1.5.0
        This is replaced by cashocs.newton_solve and will be removed in the future.
    """

    warning(
        "DEPREACTION WARNING: cashocs.damped_newton_solve is replaced by cashocs.newton_solve and will be removed in the future."
    )

    return newton_solve(
        F,
        u,
        bcs,
        dF=dF,
        shift=None,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
        convergence_type=convergence_type,
        norm_type=norm_type,
        damped=damped,
        verbose=verbose,
        ksp=ksp,
        ksp_options=ksp_options,
    )
