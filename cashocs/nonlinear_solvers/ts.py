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

"""Interface for the PETSc TS solver for pseudo time stepping."""

from __future__ import annotations

import copy
from typing import cast, TYPE_CHECKING

import fenics
import numpy as np
from petsc4py import PETSc

from cashocs import _exceptions
from cashocs import _utils
from cashocs import log

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

if TYPE_CHECKING:
    from cashocs import _typing

default_ts_pseudo_options: _typing.KspOption = {
    "ts_type": "beuler",
    "ts_dt": 1e-2,
    "ts_max_steps": 1000,
    "snes_type": "ksponly",
}


class TSPseudoSolver:
    """Interface for using PETSc's TS solver for pseudo time stepping.

    This class implements only pseudo time stepping for nonlinear equations, it should
    not be used for transient simulations.
    """

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
        excluded_from_time_derivative: list[int] | None = None,
    ) -> None:
        """Initializes the TS pseudo time stepping solver.

        Args:
            nonlinear_form (ufl.Form): The variational form of the nonlinear problem to
                be solved by Newton's method.
            u (fenics.Function): The sought solution / initial guess. It is not assumed
                that the initial guess satisfies the Dirichlet boundary conditions,
                they are applied automatically. The method overwrites / updates this
                Function.
            bcs (fenics.DirichletBC | list[fenics.DirichletBC]): A list of DirichletBCs
                for the nonlinear variational problem.
            derivative (ufl.Form | None, optional): The Jacobian of nonlinear_form,
                used for the Newton method. Default is None, and in this case the
                Jacobian is computed automatically with AD.
            petsc_options (_typing.KspOption | None, optional): The options for PETSc
                TS object. Defaults to None.
            shift (ufl.Form | None, optional): A shift term, if the right-hand side of
                the nonlinear problem is not zero, but shift. Defaults to None.
            rtol (float | None, optional): Relative tolerance of the solver. If this
                is set to a float, the float is used as relative tolerance. If this is
                set to None, then the relative tolerance of the SNES object is used,
                which can be defined with the petsc options `snes_rtol rtol`. Defaults
                to None.
            atol (float | None, optional): Absolute tolerance of the solver. If this
                is set to a float, the float is used as absolute tolerance. If this is
                set to None, then the absolute tolerance of the SNES object is used,
                which can be defined with the petsc options `snes_atol atol`. Defaults
                to None.
            max_iter (int | None, optional): Maximum number of iterations carried out
                by the method. Overrides the specification in the petsc_options.
                Defaults to None.
            A_tensor (fenics.PETScMatrix | None, optional): A fenics.PETScMatrix for
                storing the left-hand side of the linear sub-problem. Defaults to None.
            b_tensor (fenics.PETScVector | None, optional): A fenics.PETScVector for
                storing the right-hand side of the linear sub-problem. Defaults to None.
            preconditioner_form (ufl.Form | None, optional): A UFL form which defines
                the preconditioner matrix. Defaults to None.
            excluded_from_time_derivative (list[int] | None, optional): A list of
                indices for those components that are not included in the time
                derivative. Example: The pressure for incompressible Navier-Stokes.
                Default is None, so that all components are included for the time
                derivative.

        """
        self.nonlinear_form = nonlinear_form
        self.u = u

        self.space = self.u.function_space()
        self.mesh = self.space.mesh()
        self.comm = self.mesh.mpi_comm()
        self.bcs = _utils.enlist(bcs)
        self.shift = shift

        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter

        if petsc_options is None:
            self.petsc_options: _typing.KspOption = copy.deepcopy(
                default_ts_pseudo_options
            )
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
            self.A_petsc = A_tensor.mat()  # pylint: disable=invalid-name,
        else:
            self.A_petsc = fenics.PETScMatrix(self.comm).mat()

        if b_tensor is not None:
            self.residual_petsc = b_tensor.vec()
        else:
            self.residual_petsc = fenics.PETScVector(self.comm).vec()

        self.residual_convergence = fenics.PETScVector(self.comm)

        self.mass_matrix_petsc = fenics.PETScMatrix(self.comm).mat()
        self.mass_application_petsc = fenics.Function(self.space).vector().vec()

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

        self.is_preassembled = False
        if excluded_from_time_derivative is None:
            self.excluded_from_time_derivative = []
        else:
            self.excluded_from_time_derivative = excluded_from_time_derivative
        self._setup_mass_matrix()

    def _setup_mass_matrix(self) -> None:
        """Sets up the mass matrix for the time derivative."""
        space = self.u.function_space()
        trial = fenics.TrialFunction(space)
        test = fenics.TestFunction(space)

        dx = fenics.Measure("dx", domain=self.mesh)

        split_trial = fenics.split(trial)
        split_test = fenics.split(test)

        split_trial = tuple(
            fun
            for i, fun in enumerate(split_trial)
            if i not in self.excluded_from_time_derivative
        )
        split_test = tuple(
            fun
            for i, fun in enumerate(split_test)
            if i not in self.excluded_from_time_derivative
        )

        form_list = []
        for i in range(len(split_trial)):
            test_shape = split_test[i].ufl_shape
            if len(test_shape) == 1:
                mass_matrix_part = ufl.dot(split_trial[i], split_test[i]) * dx
            elif len(test_shape) == 2:
                mass_matrix_part = fenics.inner(split_trial[i], split_test[i]) * dx
            else:
                mass_matrix_part = split_trial[i] * split_test[i] * dx

            form_list.append(mass_matrix_part)

        form = _utils.summation(form_list)
        M = fenics.PETScMatrix(self.mass_matrix_petsc)  # pylint: disable=invalid-name,
        fenics.assemble(form, tensor=M)

        self.M_petsc = self.mass_matrix_petsc.copy()  # pylint: disable=invalid-name,

        if self.preconditioner_form is not None:
            self.MP_petsc = (  # pylint: disable=invalid-name,
                self.mass_matrix_petsc.copy()
            )
        else:
            self.MP_petsc = None

    def assemble_residual(
        self,
        ts: PETSc.TS,  # pylint: disable=unused-argument
        t: float,  # pylint: disable=unused-argument
        u: PETSc.Vec,
        f: PETSc.Vec,
    ) -> None:
        """Interface for PETSc TSSetRHSFunction.

        Args:
            ts (PETSc.TS): The TS solver.
            t (float): The current time step.
            u (PETSc.Vec): The current iterate.
            f (PETSc.Vec): The vector in which the residual is stored.

        """
        log.begin("Assembling the residual for pseudo time stepping.", level=log.DEBUG)
        self.u.vector().vec().aypx(0.0, u)
        self.u.vector().apply("")

        f_fenics = fenics.PETScVector(f)
        self.assembler.assemble(f_fenics, self.u.vector())

        if (
            self.shift is not None
            and self.assembler_shift is not None
            and self.residual_shift is not None
        ):
            self.assembler_shift.assemble(self.residual_shift, self.u.vector())
            f[:] -= self.residual_shift[:]

        f.scale(-1)
        log.end()

    def assemble_jacobian(
        self,
        ts: PETSc.TS,  # pylint: disable=unused-argument
        t: float,  # pylint: disable=unused-argument
        u: PETSc.Vec,
        J: PETSc.Mat,  # pylint: disable=invalid-name,
        P: PETSc.Mat,  # pylint: disable=invalid-name,
    ) -> None:
        """Interface for PETSc TSSetRHSJacobian.

        Args:
            ts (PETSc.TS): The TS solver.
            t (float): The current time step.
            u (PETSc.Vec): The current iterate.
            J (PETSc.Mat): The matrix for storing the Jacobian.
            P (PETSc.Mat): The matrix for storing the preconditioner.

        """
        log.begin("Assembling the Jacobian for pseudo time stepping.", level=log.DEBUG)
        self.u.vector().vec().aypx(0.0, u)
        self.u.vector().apply("")

        J = fenics.PETScMatrix(J)  # pylint: disable=invalid-name
        self.assembler.assemble(J)
        J.ident_zeros()

        J_petsc = fenics.as_backend_type(J).mat()  # pylint: disable=invalid-name,
        J_petsc.scale(-1)
        J_petsc.assemble()

        if self.preconditioner_form is not None:
            P = fenics.PETScMatrix(P)  # pylint: disable=invalid-name
            self.assembler.assemble(P)
            P.ident_zeros()

            P_petsc = fenics.as_backend_type(P).mat()  # pylint: disable=invalid-name,
            P_petsc.scale(-1)
            P_petsc.assemble()

        log.end()

    def assemble_i_function(
        self,
        ts: PETSc.TS,  # pylint: disable=unused-argument
        t: float,  # pylint: disable=unused-argument
        u: PETSc.Vec,  # pylint: disable=unused-argument
        u_dot: PETSc.Vec,
        f: PETSc.Vec,
    ) -> None:
        """Interface for PETSc TSSetIFunction - this describes the time derivative.

        Args:
            ts (PETSc.TS): The TS solver.
            t (float): The current time step.
            u (PETSc.Vec): The current iterate.
            u_dot (PETSc.Vec): The time derivative.
            f (PETSc.Vec): The vector for storing the IFunction.

        """
        res = self.mass_matrix_petsc.createVecLeft()
        self.mass_matrix_petsc.mult(u_dot, res)
        f.aypx(0.0, res)
        f.assemble()

    def assemble_mass_matrix(
        self,
        ts: PETSc.TS,  # pylint: disable=unused-argument
        t: float,  # pylint: disable=unused-argument
        u: PETSc.Vec,  # pylint: disable=unused-argument
        u_dot: PETSc.Vec,  # pylint: disable=unused-argument
        sigma: float,
        A: PETSc.Mat,  # pylint: disable=invalid-name,
        B: PETSc.Mat,  # pylint: disable=invalid-name,
    ) -> None:
        """Interface for PETSc  TSSetIJacobian.

        This describes the Jacobian of the time derivative. Here, the IJacobian and its
        preconditioner are given by the FEM mass matrices.

        Args:
            ts (PETSc.TS): The TS solver.
            t (float): The current time step.
            u (PETSc.Vec): The current iterate.
            u_dot (PETSc.Vec): The time derivative of the current iterate.
            sigma (float): See PETSc documentation.
            A (PETSc.Mat): The matrix for storing the IJacobian.
            B (PETSc.Mat): The matrix for storing the preconditioner of the IJacobian.

        """
        A.aypx(0.0, self.mass_matrix_petsc)
        A.scale(sigma)
        A.assemble()

        if self.preconditioner_form is not None:
            B.aypx(0.0, self.mass_matrix_petsc)
            B.scale(sigma)
            B.assemble()

    def compute_nonlinear_residual(self, u: PETSc.Vec) -> float:
        """Computes the residual of the nonlinear equation.

        Args:
            u (PETSc.Vec): The current iterate.

        Returns:
            float: The norm of the nonlinear residual.

        """
        self.u.vector().vec().aypx(0.0, u)
        self.u.vector().apply("")
        self.assembler.assemble(self.residual_convergence, self.u.vector())

        residual_norm: float = self.residual_convergence.norm("l2")
        return residual_norm

    def monitor(
        self,
        ts: PETSc.TS,
        i: int,
        t: float,
        u: PETSc.Vec,  # pylint: disable=unused-argument
    ) -> None:
        """The monitoring function for the pseudo time stepping.

        Args:
            ts (PETSc.TS): The TS solver.
            i (int): The current iteration number.
            t (float): The current time step.
            u (PETSc.Vec): The current iterate.

        """
        residual_norm = self.compute_nonlinear_residual(u)

        log.debug(f"TS {i = }  {t = :.3e}  residual: {residual_norm:.3e}")

        self.rtol = cast(float, self.rtol)
        self.atol = cast(float, self.atol)

        if residual_norm < np.maximum(self.rtol * self.res_initial, self.atol):
            ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)
            max_time = ts.getMaxTime()
            ts.setTime(max_time)

    def solve(self) -> fenics.Function:
        """Solves the (nonlinear) problem with pseudo time stepping.

        Returns:
            fenics.Function: The solution obtained by the solver.

        """
        log.begin("Solving the PDE system with pseudo time stepping.")
        ts = PETSc.TS().create()
        ts.setProblemType(ts.ProblemType.NONLINEAR)

        ts.setIFunction(self.assemble_i_function, self.mass_application_petsc)
        ts.setIJacobian(self.assemble_mass_matrix, self.M_petsc, self.MP_petsc)
        ts.setRHSFunction(self.assemble_residual, self.residual_petsc)
        ts.setRHSJacobian(self.assemble_jacobian, self.A_petsc, self.P_petsc)

        ksp = ts.getSNES().getKSP()
        _utils.linalg.setup_fieldsplit_preconditioner(self.u, ksp, self.petsc_options)

        if fenics.PETScMatrix(self.A_petsc).empty():
            self.assemble_i_function(
                ts,
                0.0,
                None,
                self.u.vector().vec(),
                self.mass_application_petsc,
            )
            self.assemble_mass_matrix(
                ts,
                0.0,
                self.u.vector().vec(),
                self.u.vector().vec(),
                0.0,
                self.M_petsc,
                self.MP_petsc,
            )
            self.assemble_residual(ts, 0.0, self.u.vector().vec(), self.residual_petsc)
            self.assemble_jacobian(
                ts, 0.0, self.u.vector().vec(), self.A_petsc, self.P_petsc
            )

        self.res_initial = self.compute_nonlinear_residual(self.u.vector().vec())
        ts.setTime(0.0)
        ts.setMonitor(self.monitor)

        _utils.setup_petsc_options([ts], [self.petsc_options])
        if self.rtol is None:
            self.rtol = ts.getSNES().rtol
        if self.atol is None:
            self.atol = ts.getSNES().atol
        if self.max_iter is not None:
            ts.setMaxSteps(self.max_iter)

        x = fenics.Function(self.space)
        x.vector().vec().aypx(0.0, self.u.vector().vec())
        x.vector().apply("")

        ts.solve(x.vector().vec())

        self.u.vector().vec().aypx(0.0, x.vector().vec())
        self.u.vector().apply("")

        converged_reason = ts.getConvergedReason()
        if (
            converged_reason < 0
            or converged_reason == PETSc.TS.ConvergedReason.CONVERGED_ITS
        ):
            raise _exceptions.PETScTSError(converged_reason)

        if hasattr(PETSc, "garbage_cleanup"):
            ts.destroy()
            PETSc.garbage_cleanup(comm=self.comm)
            PETSc.garbage_cleanup()

        log.end()

        return self.u


def ts_pseudo_solve(
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
    excluded_from_time_derivative: list[int] | None = None,
) -> fenics.Function:
    """Solve a nonlinear PDE problem with PETSc TS and pseudo time stepping.

    An overview over possible PETSc command line options for the TS can be found
    at `<https://petsc.org/release/manualpages/TS/>`_

    Args:
        nonlinear_form (ufl.Form): The variational form of the nonlinear problem to
            be solved by Newton's method.
        u (fenics.Function): The sought solution / initial guess. It is not assumed
            that the initial guess satisfies the Dirichlet boundary conditions,
            they are applied automatically. The method overwrites / updates this
            Function.
        bcs (fenics.DirichletBC | list[fenics.DirichletBC]): A list of DirichletBCs
            for the nonlinear variational problem.
        derivative (ufl.Form | None, optional): The Jacobian of nonlinear_form,
            used for the Newton method. Default is None, and in this case the
            Jacobian is computed automatically with AD.
        petsc_options (_typing.KspOption | None, optional): The options for PETSc
            TS object. Defaults to None.
        shift (ufl.Form | None, optional): A shift term, if the right-hand side of
            the nonlinear problem is not zero, but shift. Defaults to None.
        rtol (float | None, optional): Relative tolerance of the solver. If this
            is set to a float, the float is used as relative tolerance. If this is
            set to None, then the relative tolerance of the SNES object is used, which
            can be defined with the petsc options `snes_rtol rtol`. Defaults to None.
        atol (float | None, optional): Absolute tolerance of the solver. If this
            is set to a float, the float is used as absolute tolerance. If this is
            set to None, then the absolute tolerance of the SNES object is used, which
            can be defined with the petsc options `snes_atol atol`. Defaults to None.
        max_iter (int | None, optional): Maximum number of iterations carried out
            by the method. Overrides the specification in the petsc_options.
            Defaults to None.
        A_tensor (fenics.PETScMatrix | None, optional): A fenics.PETScMatrix for
            storing the left-hand side of the linear sub-problem. Defaults to None.
        b_tensor (fenics.PETScVector | None, optional): A fenics.PETScVector for
            storing the right-hand side of the linear sub-problem. Defaults to None.
        preconditioner_form (ufl.Form | None, optional): A UFL form which defines
            the preconditioner matrix. Defaults to None.
        excluded_from_time_derivative (list[int] | None, optional): A list of
            indices for those components that are not included in the time
            derivative. Example: The pressure for incompressible Navier-Stokes.
            Default is None, so that all components are included for the time
            derivative.

    Returns:
        The solution in form of a FEniCS Function.

    """
    solver = TSPseudoSolver(
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
        excluded_from_time_derivative=excluded_from_time_derivative,
    )

    solution = solver.solve()
    return solution
