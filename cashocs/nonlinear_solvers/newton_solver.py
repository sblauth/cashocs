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

import copy
from typing import List, Optional, Union

import fenics
import numpy as np
from typing_extensions import Literal
import ufl

from cashocs import _exceptions
from cashocs import _utils


class _NewtonSolver:
    """A Newton solver."""

    def __init__(
        self,
        nonlinear_form: ufl.Form,
        u: fenics.Function,
        bcs: Union[fenics.DirichletBC, List[fenics.DirichletBC]],
        derivative: Optional[ufl.Form] = None,
        shift: Optional[ufl.Form] = None,
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_iter: int = 50,
        convergence_type: Literal["combined", "rel", "abs"] = "combined",
        norm_type: Literal["l2", "linf"] = "l2",
        damped: bool = True,
        inexact: bool = True,
        verbose: bool = True,
        ksp_options: Optional[List[List[Union[str, int, float]]]] = None,
        A_tensor: Optional[fenics.PETScMatrix] = None,  # pylint: disable=invalid-name
        b_tensor: Optional[fenics.PETScVector] = None,
        is_linear: bool = False,
    ) -> None:
        r"""Initializes self.

        Args:
            nonlinear_form: The variational form of the nonlinear problem to be solved
                by Newton's method.
            u: The sought solution / initial guess. It is not assumed that the initial
                guess satisfies the Dirichlet boundary conditions, they are applied
                automatically. The method overwrites / updates this Function.
            bcs: A list of DirichletBCs for the nonlinear variational problem.
            derivative: The Jacobian of nonlinear_form, used for the Newton method.
                Default is None, and in this case the Jacobian is computed automatically
                with AD.
            shift: A shift term, if the right-hand side of the nonlinear problem is not
                zero, but shift.
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
            inexact: A boolean flag which indicates, whether an inexact Newton\'s method
                is used.
            verbose: If ``True``, prints status of the iteration to the console (default
                is ``True``).
            ksp_options: The list of options for the linear solver.
            A_tensor: A fenics.PETScMatrix for storing the left-hand side of the linear
                sub-problem.
            b_tensor: A fenics.PETScVector for storing the right-hand side of the linear
                sub-problem.
            is_linear: A boolean flag, which indicates whether the problem is actually
                linear.

        """
        self.nonlinear_form = nonlinear_form
        self.u = u
        self.bcs = _utils.enlist(bcs)
        self.shift = shift
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.convergence_type = convergence_type
        self.norm_type = norm_type
        self.damped = damped
        self.inexact = inexact
        self.A_tensor = A_tensor  # pylint: disable=invalid-name
        self.b_tensor = b_tensor
        self.is_linear = is_linear

        self.verbose = verbose if not self.is_linear else False

        temp_derivative = derivative or fenics.derivative(self.nonlinear_form, self.u)
        self.derivative = _utils.bilinear_boundary_form_modification([temp_derivative])[
            0
        ]

        # Setup increment and function for monotonicity test
        self.function_space = u.function_space()
        self.du = fenics.Function(self.function_space)
        self.ddu = fenics.Function(self.function_space)
        self.u_save = fenics.Function(self.function_space)
        self.ksp_options = ksp_options

        if ksp_options is None:
            self.ksp_options = copy.deepcopy(_utils.linalg.direct_ksp_options)
        else:
            self.ksp_options = ksp_options

        self.iterations = 0
        for bc in self.bcs:
            bc.apply(self.u.vector())
        # copy the boundary conditions and homogenize them for the increment
        self.bcs_hom = [fenics.DirichletBC(bc) for bc in self.bcs]
        for bc in self.bcs_hom:
            bc.homogenize()

        # inexact newton parameters
        self.eta = 1.0
        self.eta_max = 0.9999
        self.eta_a = 1.0
        self.gamma = 0.9
        self.lmbd = 1.0

        self.assembler = fenics.SystemAssembler(
            self.derivative, -self.nonlinear_form, self.bcs_hom
        )
        self.assembler.keep_diagonal = True
        # pylint: disable=invalid-name
        self.A_fenics = self.A_tensor or fenics.PETScMatrix()
        self.residual = self.b_tensor or fenics.PETScVector()
        self.b = fenics.as_backend_type(self.residual).vec()
        self.A_matrix = fenics.as_backend_type(self.A_fenics).mat()

        self.assembler_shift: Optional[fenics.SystemAssembler] = None
        self.residual_shift: Optional[fenics.PETScVector] = None
        if self.shift is not None:
            self.assembler_shift = fenics.SystemAssembler(
                self.derivative, self.shift, self.bcs_hom
            )
            self.residual_shift = fenics.PETScVector()

        self.breakdown = False
        self.res = 1.0
        self.res_0 = 1.0
        self.tol = 1.0

    def _print_output(self) -> None:
        """Prints the output of the current iteration to the console."""
        if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            prefix = "Newton solver:  "

            if self.iterations % 10 == 0:
                info_str = (
                    f"\n{prefix}iter,  "
                    f"abs. residual (abs. tol),  "
                    f"rel. residual (rel. tol)\n\n"
                )
            else:
                info_str = ""

            print_str = (
                f"{prefix}{self.iterations:4d},  "
                f"{self.res:>13.3e} ({self.atol:.2e}),  "
                f"{self.res/self.res_0:>13.3e} ({self.rtol:.2e})"
            )

            print(info_str + print_str, flush=True)
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def _assemble_matrix(self) -> None:
        """Assembles the matrix for solving the linear problem."""
        self.assembler.assemble(self.A_fenics)
        self.A_fenics.ident_zeros()
        self.A_matrix = fenics.as_backend_type(  # pylint: disable=invalid-name
            self.A_fenics
        ).mat()

    def _compute_eta_inexact(self) -> None:
        """Computes the parameter ``eta`` for the inexact Newton method."""
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

    def _check_for_nan_residual(self) -> None:
        """Checks, whether the residual is nan. If yes, raise a NotConvergedError."""
        if np.isnan(self.res):
            raise _exceptions.NotConvergedError("newton solver", "Residual is nan.")

    def solve(self) -> fenics.Function:
        r"""Solves the (nonlinear) problem with Newton\'s method.

        Returns:
            A solution of the nonlinear problem.

        """
        self._compute_residual()

        self.res_0 = self.residual.norm(self.norm_type)
        if self.res_0 == 0.0:  # pragma: no cover
            if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                print("Residual vanishes, input is already a solution.", flush=True)
            fenics.MPI.barrier(fenics.MPI.comm_world)
            return self.u

        self.res = self.res_0
        self._check_for_nan_residual()
        self._print_output()

        self.tol = self._compute_convergence_tolerance()

        # While loop until termination
        while self.res > self.tol and self.iterations < self.max_iter:
            self._assemble_matrix()

            self.iterations += 1
            self.lmbd = 1.0
            self.breakdown = False
            self.u_save.vector().vec().aypx(0.0, self.u.vector().vec())
            self.u_save.vector().apply("")

            self._compute_eta_inexact()
            _utils.solve_linear_problem(
                A=self.A_matrix,
                b=self.b,
                x=self.du.vector().vec(),
                ksp_options=self.ksp_options,
                rtol=self.eta,
                atol=self.atol / 10.0,
            )
            self.du.vector().apply("")

            if self.is_linear:
                self.u.vector().vec().axpy(1.0, self.du.vector().vec())
                self.u.vector().apply("")
                break

            self._backtracking_line_search()
            self._compute_residual()

            for bc in self.bcs_hom:
                bc.apply(self.residual)

            res_prev = self.res
            self.res = self.residual.norm(self.norm_type)
            self._check_for_nan_residual()
            self.eta_a = self.gamma * pow(self.res / res_prev, 2)
            self._print_output()

            if self._check_for_convergence():
                break

        self._check_if_successful()

        return self.u

    def _check_for_convergence(self) -> bool:
        """Checks, whether the desired convergence tolerance has been reached."""
        if self.res <= self.tol:
            if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                print(
                    f"\nNewton Solver converged "
                    f"after {self.iterations:d} iterations.\n",
                    flush=True,
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)
            return True

        else:
            return False

    def _check_if_successful(self) -> None:
        """Checks, whether the attempted solve was successful."""
        if self.res > self.tol and not self.is_linear:
            raise _exceptions.NotConvergedError(
                "Newton solver",
                f"The Newton solver did not converge after "
                f"{self.iterations:d} iterations.",
            )

    def _check_for_divergence(self) -> None:
        """Checks, whether the Newton solver diverged."""
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
        """Computes the residual of the nonlinear system."""
        self.assembler.assemble(self.residual)
        if (
            self.shift is not None
            and self.assembler_shift is not None
            and self.residual_shift is not None
        ):
            self.assembler_shift.assemble(self.residual_shift)
            self.residual[:] += self.residual_shift[:]

        self.b = fenics.as_backend_type(self.residual).vec()

    def _compute_convergence_tolerance(self) -> float:
        """Computes the tolerance for the Newton solver.

        Returns:
            The computed tolerance.

        """
        if self.convergence_type.casefold() == "abs":
            return self.atol
        elif self.convergence_type.casefold() == "rel":
            return self.rtol * self.res_0
        else:
            return self.rtol * self.res_0 + self.atol

    def _backtracking_line_search(self) -> None:
        """Performs a backtracking line search for the damped Newton method."""
        if self.damped:
            while True:
                self.u.vector().vec().axpy(self.lmbd, self.du.vector().vec())
                self.u.vector().apply("")
                self._compute_residual()
                _utils.solve_linear_problem(
                    A=self.A_matrix,
                    b=self.b,
                    x=self.ddu.vector().vec(),
                    ksp_options=self.ksp_options,
                    rtol=self.eta,
                    atol=self.atol / 10.0,
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
                    self.u.vector().apply("")
                    self.lmbd /= 2

                if self.lmbd < 1e-6:
                    self.breakdown = True
                    break
        else:
            self.u.vector().vec().axpy(1.0, self.du.vector().vec())
            self.u.vector().apply("")


def newton_solve(
    nonlinear_form: ufl.Form,
    u: fenics.Function,
    bcs: Union[fenics.DirichletBC, List[fenics.DirichletBC]],
    derivative: Optional[ufl.Form] = None,
    shift: Optional[ufl.Form] = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_iter: int = 50,
    convergence_type: Literal["combined", "rel", "abs"] = "combined",
    norm_type: Literal["l2", "linf"] = "l2",
    damped: bool = True,
    inexact: bool = True,
    verbose: bool = True,
    ksp_options: Optional[List[List[Union[str, int, float]]]] = None,
    A_tensor: Optional[fenics.PETScMatrix] = None,  # pylint: disable=invalid-name
    b_tensor: Optional[fenics.PETScVector] = None,
    is_linear: bool = False,
) -> fenics.Function:
    r"""Solves a nonlinear problem with Newton\'s method.

    Args:
        nonlinear_form: The variational form of the nonlinear problem to be solved by
            Newton's method.
        u: The sought solution / initial guess. It is not assumed that the initial guess
            satisfies the Dirichlet boundary conditions, they are applied automatically.
            The method overwrites / updates this Function.
        bcs: A list of DirichletBCs for the nonlinear variational problem.
        derivative: The Jacobian of nonlinear_form, used for the Newton method. Default
            is None, and in this case the Jacobian is computed automatically with AD.
        shift: A shift term, if the right-hand side of the nonlinear problem is not
            zero, but shift.
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
        inexact: If ``True``, an inexact Newton\'s method is used. Default is ``True``.
        verbose: If ``True``, prints status of the iteration to the console (default is
            ``True``).
        ksp_options: The list of options for the linear solver.
        A_tensor: A fenics.PETScMatrix for storing the left-hand side of the linear
            sub-problem.
        b_tensor: A fenics.PETScVector for storing the right-hand side of the linear
            sub-problem.
        is_linear: A boolean flag, which indicates whether the problem is actually
            linear.

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

            u = Function(function_space)
            v = TestFunction(function_space)
            F = inner(grad(u), grad(v))*dx + pow(u,3)*v*dx - Constant(1)*v*dx
            bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1,2,3,4])
            cashocs.newton_solve(F, u, bcs)

    """
    solver = _NewtonSolver(
        nonlinear_form,
        u,
        bcs,
        derivative=derivative,
        shift=shift,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
        convergence_type=convergence_type,
        norm_type=norm_type,
        damped=damped,
        inexact=inexact,
        verbose=verbose,
        ksp_options=ksp_options,
        A_tensor=A_tensor,
        b_tensor=b_tensor,
        is_linear=is_linear,
    )

    solution = solver.solve()
    return solution
