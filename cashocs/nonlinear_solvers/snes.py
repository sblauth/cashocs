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

"""Interface for the PETSc SNES solver for nonlinear equations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import fenics
from petsc4py import PETSc

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs import log

if TYPE_CHECKING:
    from cashocs import _typing


default_snes_options: _typing.KspOption = {
    "snes_type": "newtonls",
    "snes_atol": 1e-10,
    "snes_rtol": 1e-10,
    "snes_max_it": 50,
}


class SNESSolver:
    """Interface for using PETSc's SNES solver."""

    def __init__(
        self,
        nonlinear_form: ufl.Form,
        u: fenics.Function,
        bcs: fenics.DirichletBC | list[fenics.DirichletBC],
        derivative: ufl.Form | None = None,
        petsc_options: _typing.KspOption | None = None,
        shift: ufl.Form | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_iter: int | None = None,
        A_tensor: fenics.PETScMatrix | None = None,  # pylint: disable=invalid-name
        b_tensor: fenics.PETScVector | None = None,
        preconditioner_form: ufl.Form | None = None,
    ) -> None:
        """Initialize the SNES solver.

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
            petsc_options: The options for PETSc SNES object.
            shift: A shift term, if the right-hand side of the nonlinear problem is not
                zero, but shift.
            rtol: Relative tolerance of the solver. Overrides the specification in the
                petsc_options.
            atol: Absolute tolerance of the solver. Overrides the specification in the
                petsc_options.
            max_iter: Maximum number of iterations carried out by the method. Overrides
                the specification in the petsc_options.
            A_tensor: A fenics.PETScMatrix for storing the left-hand side of the linear
                sub-problem.
            b_tensor: A fenics.PETScVector for storing the right-hand side of the linear
                sub-problem.
            preconditioner_form: A UFL form which defines the preconditioner matrix.

        """
        self.nonlinear_form = nonlinear_form
        self.u = u
        self.comm = self.u.function_space().mesh().mpi_comm()
        self.bcs = _utils.enlist(bcs)
        self.shift = shift

        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter

        self.is_preassembled = False

        if petsc_options is None:
            self.petsc_options: _typing.KspOption = copy.deepcopy(default_snes_options)
            self.petsc_options.update(_utils.linalg.direct_ksp_options)
        else:
            self.petsc_options = petsc_options

        if preconditioner_form is not None:
            if len(preconditioner_form.arguments()) == 1:
                self.preconditioner_form = fenics.derivative(
                    preconditioner_form, self.u
                )
            else:
                self.preconditioner_form = preconditioner_form
        else:
            self.preconditioner_form = None

        temp_derivative = derivative or fenics.derivative(self.nonlinear_form, self.u)
        self.derivative = _utils.bilinear_boundary_form_modification([temp_derivative])[
            0
        ]

        self.assembler = fenics.SystemAssembler(
            self.derivative, self.nonlinear_form, self.bcs
        )
        self.assembler.keep_diagonal = True

        if A_tensor is not None:
            self.A_petsc = A_tensor.mat()  # pylint: disable=invalid-name
        else:
            self.A_petsc = fenics.PETScMatrix(self.comm).mat()

        if b_tensor is not None:
            self.residual_petsc = b_tensor.vec()
        else:
            self.residual_petsc = fenics.PETScVector(self.comm).vec()

        if self.preconditioner_form is not None:
            self.assembler_pc = fenics.SystemAssembler(
                self.preconditioner_form, self.nonlinear_form, self.bcs
            )
            self.assembler_pc.keep_diagonal = True
            self.P_petsc = fenics.PETScMatrix(  # pylint: disable=invalid-name
                self.comm
            ).mat()
        else:
            self.P_petsc = None

        self.assembler_shift: fenics.SystemAssembler | None = None
        self.residual_shift: fenics.PETScVector | None = None
        if self.shift is not None:
            self.assembler_shift = fenics.SystemAssembler(
                self.derivative, self.shift, self.bcs
            )
            self.residual_shift = fenics.PETScVector(self.comm)

    @log.profile_execution_time("assembling the residual for SNES")
    def assemble_function(
        self,
        snes: PETSc.SNES,  # pylint: disable=unused-argument
        x: PETSc.Vec,
        f: PETSc.Vec,
    ) -> None:
        """Interface for PETSc SNESSetFunction.

        Args:
            snes: The SNES solver
            x: The current iterate
            f: The vector in which the function evaluation is stored.

        """
        self.u.vector().vec().aypx(0.0, x)
        self.u.vector().apply("")
        f = fenics.PETScVector(f)

        self.assembler.assemble(f, self.u.vector())
        if (
            self.shift is not None
            and self.assembler_shift is not None
            and self.residual_shift is not None
        ):
            self.assembler_shift.assemble(self.residual_shift, self.u.vector())
            f[:] -= self.residual_shift[:]

    @log.profile_execution_time("assembling the Jacobian for SNES")
    def assemble_jacobian(
        self,
        snes: PETSc.SNES,  # pylint: disable=unused-argument
        x: PETSc.Vec,
        J: PETSc.Mat,  # pylint: disable=invalid-name,
        P: PETSc.Mat,  # pylint: disable=invalid-name
    ) -> None:
        """Interface for PETSc SNESSetJacobian.

        Args:
            snes: The SNES solver.
            x: The current iterate.
            J: The matrix storing the Jacobian.
            P: The matrix storing the preconditioner for the Jacobian.

        """
        if not self.is_preassembled:
            self.u.vector().vec().aypx(0.0, x)
            self.u.vector().apply("")

            J = fenics.PETScMatrix(J)  # pylint: disable=invalid-name
            self.assembler.assemble(J)
            J.ident_zeros()

            if self.preconditioner_form is not None:
                P = fenics.PETScMatrix(P)  # pylint: disable=invalid-name
                self.assembler_pc.assemble(P)
                P.ident_zeros()
        else:
            self.is_preassembled = False

    def solve(self) -> fenics.Function:
        """Solves the nonlinear problem with PETSc's SNES."""
        log.begin("Solving the nonlinear PDE system with PETSc SNES.", level=log.DEBUG)
        snes = PETSc.SNES().create(self.comm)

        snes.setFunction(self.assemble_function, self.residual_petsc)
        snes.setJacobian(self.assemble_jacobian, self.A_petsc, self.P_petsc)

        ksp = snes.getKSP()
        _utils.linalg.setup_fieldsplit_preconditioner(self.u, ksp, self.petsc_options)

        if fenics.PETScMatrix(self.A_petsc).empty():
            self.assemble_jacobian(
                snes, self.u.vector().vec(), self.A_petsc, self.P_petsc
            )
            self.assemble_function(snes, self.u.vector().vec(), self.residual_petsc)
            self.is_preassembled = True

        _utils.setup_petsc_options([snes], [self.petsc_options])
        snes.setTolerances(rtol=self.rtol, atol=self.atol, max_it=self.max_iter)

        x = fenics.Function(self.u.function_space())
        x.vector().vec().aypx(0.0, self.u.vector().vec())
        x.vector().apply("")

        snes.solve(None, x.vector().vec())
        converged_reason = snes.getConvergedReason()

        if hasattr(PETSc, "garbage_cleanup"):
            snes.destroy()
            PETSc.garbage_cleanup(comm=self.comm)

        self.u.vector().vec().setArray(x.vector().vec())
        self.u.vector().apply("")

        log.end()

        if converged_reason < 0:
            raise _exceptions.PETScSNESError(converged_reason)

        return self.u


def snes_solve(
    nonlinear_form: ufl.Form,
    u: fenics.Function,
    bcs: fenics.DirichletBC | list[fenics.DirichletBC],
    derivative: ufl.Form | None = None,
    petsc_options: _typing.KspOption | None = None,
    shift: ufl.Form | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int | None = None,
    A_tensor: fenics.PETScMatrix | None = None,  # pylint: disable=invalid-name
    b_tensor: fenics.PETScVector | None = None,
    preconditioner_form: ufl.Form | None = None,
) -> fenics.Function:
    """Solve a nonlinear PDE problem with PETSc SNES.

    An overview over possible PETSc command line options for the SNES can be found
    at `<https://petsc.org/release/manualpages/SNES/>`_.

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
        petsc_options: The options for PETSc SNES object.
        shift: A shift term, if the right-hand side of the nonlinear problem is not
            zero, but shift.
        rtol: Relative tolerance of the solver (default is ``rtol = 1e-10``).
        atol: Absolute tolerance of the solver (default is ``atol = 1e-10``).
        max_iter: Maximum number of iterations carried out by the method (default is
            ``max_iter = 50``).
        A_tensor: A fenics.PETScMatrix for storing the left-hand side of the linear
            sub-problem.
        b_tensor: A fenics.PETScVector for storing the right-hand side of the linear
            sub-problem.
        preconditioner_form: A UFL form which defines the preconditioner matrix.

    Returns:
        The solution in form of a FEniCS Function.

    """
    solver = SNESSolver(
        nonlinear_form,
        u,
        bcs,
        derivative=derivative,
        petsc_options=petsc_options,
        shift=shift,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
        A_tensor=A_tensor,
        b_tensor=b_tensor,
        preconditioner_form=preconditioner_form,
    )

    solution = solver.solve()
    return solution
