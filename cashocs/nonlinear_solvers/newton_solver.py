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

"""Newton solver for nonlinear PDEs."""

from __future__ import annotations

from typing import List, Union, Optional

import fenics
import numpy as np
import ufl
from petsc4py import PETSc
from typing_extensions import Literal

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import utils


class _NewtonSolver:
    # noinspection PyPep8Naming,PyUnresolvedReferences
    def __init__(
        self,
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
        is_linear: bool = False,
    ) -> None:

        self.F = F
        self.u = u
        if isinstance(bcs, fenics.DirichletBC):
            self.bcs = [bcs]
        else:
            self.bcs = bcs
        self.shift = shift
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.convergence_type = convergence_type
        self.norm_type = norm_type
        self.damped = damped
        self.inexact = inexact
        self.verbose = verbose
        self.A_tensor = A_tensor
        self.b_tensor = b_tensor
        self.is_linear = is_linear

        self.dF = dF or fenics.derivative(self.F, self.u)

        # Setup increment and function for monotonicity test
        self.V = u.function_space()
        self.du = fenics.Function(self.V)
        self.ddu = fenics.Function(self.V)
        self.u_save = fenics.Function(self.V)
        self.ksp_options = ksp_options

        if ksp is None:
            if ksp_options is None:
                self.ksp_options = [
                    ["ksp_type", "preonly"],
                    ["pc_type", "lu"],
                    ["pc_factor_mat_solver_type", "mumps"],
                    ["mat_mumps_icntl_24", 1],
                ]
            else:
                self.ksp_options = ksp_options

            self.ksp = PETSc.KSP().create()
            utils._setup_petsc_options([self.ksp], [self.ksp_options])

        else:
            self.ksp = ksp

        self.iterations = 0
        [bc.apply(self.u.vector()) for bc in self.bcs]
        # copy the boundary conditions and homogenize them for the increment
        self.bcs_hom = [fenics.DirichletBC(bc) for bc in self.bcs]
        [bc.homogenize() for bc in self.bcs_hom]

        # inexact newton parameters
        self.eta = 1.0
        self.eta_max = 0.9999
        self.eta_a = 1.0
        self.gamma = 0.9
        self.lmbd = 1.0

        self.assembler = fenics.SystemAssembler(self.dF, -self.F, self.bcs_hom)
        self.assembler.keep_diagonal = True
        self.A_fenics = self.A_tensor or fenics.PETScMatrix()
        self.residual = self.b_tensor or fenics.PETScVector()

        self.assembler_shift = None
        self.residual_shift = None

        if self.shift is not None:
            self.assembler_shift = fenics.SystemAssembler(
                self.dF, self.shift, self.bcs_hom
            )
            self.residual_shift = fenics.PETScVector()

        self.b = None
        self.A = None
        self.breakdown = False
        self.res = 1.0
        self.res_0 = 1.0
        self.tol = 1.0

    def _print_output(self) -> None:

        if self.verbose:
            print(
                f"Newton Iteration {self.iterations:2d} - "
                f"residual (abs):  {self.res:.3e} (tol = {self.atol:.3e})    "
                f"residual (rel):  {self.res / self.res_0:.3e} (tol = {self.rtol:.3e})"
            )

    def _assemble_matrix(self) -> None:
        self.assembler.assemble(self.A_fenics)
        self.A_fenics.ident_zeros()
        self.A = fenics.as_backend_type(self.A_fenics).mat()

    def _compute_eta_inexact(self) -> None:
        if self.inexact:
            if self.iterations == 1:
                self.eta = self.eta_max
            elif self.gamma * pow(self.eta, 2) <= 0.1:
                self.eta = np.minimum(self.eta_max, self.eta_a)
            else:
                self.eta = np.minimum(
                    self.eta_max,
                    np.maximum(self.eta_a, self.gamma * pow(self.eta, 2)),
                )

            self.eta = np.minimum(
                self.eta_max,
                np.maximum(self.eta, 0.5 * self.tol / self.res),
            )
        else:
            self.eta = self.rtol * 1e-1

        if self.is_linear:
            self.eta = self.rtol * 1e-1

    def solve(self) -> fenics.Function:
        self._compute_residual()

        self.res_0 = self.residual.norm(self.norm_type)
        if self.res_0 == 0.0:  # pragma: no cover
            if self.verbose:
                print("Residual vanishes, input is already a solution.")
            return self.u

        self.res = self.res_0
        self._print_output()

        self.tol = self._compute_convergence_tolerance()

        # While loop until termination
        while self.res > self.tol and self.iterations < self.max_iter:
            self._assemble_matrix()

            self.iterations += 1
            self.lmbd = 1.0
            self.breakdown = False
            self.u_save.vector().vec().aypx(0.0, self.u.vector().vec())

            self._compute_eta_inexact()
            utils._solve_linear_problem(
                self.ksp,
                self.A,
                self.b,
                self.du.vector().vec(),
                self.ksp_options,
                rtol=self.eta,
            )
            self.du.vector().apply("")

            if self.is_linear:
                self.u.vector().vec().axpy(1.0, self.du.vector().vec())
                break

            if self.damped:
                self._backtracking_line_search()
            else:
                self.u.vector().vec().axpy(1.0, self.du.vector().vec())

            self._compute_residual()

            [bc.apply(self.residual) for bc in self.bcs_hom]

            res_prev = self.res
            self.res = self.residual.norm(self.norm_type)
            self.eta_a = self.gamma * pow(self.res / res_prev, 2)
            self._print_output()

            if self.res <= self.tol:
                if self.verbose:
                    print(
                        f"\nNewton Solver converged "
                        f"after {self.iterations:d} iterations.\n"
                    )
                break

        return self.u

    def _check_for_divergence(self) -> None:
        if self.breakdown:
            raise _exceptions.NotConvergedError(
                "Newton solver", "Stepsize for increment too low."
            )

        if self.iterations == self.max_iter:
            raise _exceptions.NotConvergedError(
                "Newton solver",
                "Maximum number of iterations were exceeded.",
            )

    def _compute_residual(self) -> None:
        self.assembler.assemble(self.residual)
        if self.shift is not None:
            self.assembler_shift.assemble(self.residual_shift)
            self.residual[:] += self.residual_shift[:]

        self.b = fenics.as_backend_type(self.residual).vec()

    def _compute_convergence_tolerance(self) -> float:
        if self.convergence_type.casefold() == "abs":
            return self.atol
        elif self.convergence_type.casefold() == "rel":
            return self.rtol * self.res_0
        else:
            return self.rtol * self.res_0 + self.atol

    def _backtracking_line_search(self) -> None:
        while True:
            self.u.vector().vec().axpy(self.lmbd, self.du.vector().vec())
            self._compute_residual()
            utils._solve_linear_problem(
                ksp=self.ksp,
                b=self.b,
                x=self.ddu.vector().vec(),
                ksp_options=self.ksp_options,
                rtol=self.eta,
            )
            self.ddu.vector().apply("")

            if (
                self.ddu.vector().norm(self.norm_type)
                / self.du.vector().norm(self.norm_type)
                <= 1
            ):
                break
            else:
                self.u.vector().vec().aypx(0.0, self.u_save.vector().vec())
                self.lmbd /= 2

            if self.lmbd < 1e-6:
                self.breakdown = True
                break


# noinspection PyPep8Naming,PyUnresolvedReferences
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
    is_linear: bool = False,
) -> fenics.Function:
    solver = _NewtonSolver(
        F,
        u,
        bcs,
        dF=dF,
        shift=shift,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
        convergence_type=convergence_type,
        norm_type=norm_type,
        damped=damped,
        inexact=inexact,
        verbose=verbose,
        ksp=ksp,
        ksp_options=ksp_options,
        A_tensor=A_tensor,
        b_tensor=b_tensor,
        is_linear=is_linear,
    )

    solution = solver.solve()
    return solution


# noinspection PyPep8Naming,PyUnresolvedReferences
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
) -> fenics.Function:  # pragma: no cover
    r"""Damped Newton solve interface, only here for compatibility reasons.

    Args:
        F: The variational form of the nonlinear problem to be solved by Newton's
            method.
        u: The sought solution / initial guess. It is not assumed that the initial guess
            satisfies the Dirichlet boundary conditions, they are applied automatically.
            The method overwrites / updates this Function.
        bcs: A list of DirichletBCs for the nonlinear variational problem.
        dF: The Jacobian of F, used for the Newton method. Default is None, and in this
            case the Jacobian is computed automatically with AD.
        rtol: Relative tolerance of the solver if convergence_type is either
            ``'combined'`` or ``'rel'`` (default is ``rtol = 1e-10``).
        atol: Absolute tolerance of the solver if convergence_type is either
            ``'combined'`` or ``'abs'`` (default is ``atol = 1e-10``).
        max_iter: Maximum number of iterations carried out by the method (default is
            ``max_iter = 50``).
        convergence_type: Determines the type of stopping criterion that is used.
        norm_type: Determines which norm is used in the stopping criterion.
        damped: If ``True``, then a damping strategy is used. If ``False``, the
            classical Newton-Raphson iteration (without damping) is used (default is
            ``True``).
        verbose: If ``True``, prints status of the iteration to the console (default is
            ``True``).
        ksp: The PETSc ksp object used to solve the inner (linear) problem if this is
            ``None`` it uses the direct solver MUMPS (default is ``None``).
        ksp_options: The list of options for the linear solver.

    Returns:
        The solution of the nonlinear variational problem, if converged. This overwrites
        the input function u.

    Examples:
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
            cashocs.newton_solve(F, u, bcs)

    .. deprecated:: 1.5.0
        This is replaced by cashocs.newton_solve and will be removed in the future.
    """

    _loggers.warning(
        "DEPREACTION WARNING: cashocs.damped_newton_solve is replaced "
        "by cashocs.newton_solve and will be removed in the future."
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
