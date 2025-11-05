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

from fenics import *
from petsc4py import PETSc

import cashocs

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(16)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
v = TestFunction(V)

F = dot(grad(u), grad(v)) * dx - Constant(1.0) * v * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])

ksp_options = {
    "ksp_type": "cg",
    "pc_type": "jacobi",
    "ksp_monitor_true_residual": None,
    "ksp_view": None,
}
# Solve with default petsc PC and show KSP monitor
cashocs.linear_solve(F, u, bcs, ksp_options=ksp_options)

# reset u
u.vector().vec().set(0.0)
u.vector().apply("")


# Use the example Jacobi PC from petsc4py
class JacobiPC:
    # Setup the internal data. In this case, we access the matrix diagonal.
    def setUp(self, pc):
        _, P = pc.getOperators()
        self.D = P.getDiagonal()

    # Apply the preconditioner
    def apply(self, pc, x, y):
        y.pointwiseDivide(x, self.D)


# Setup the custom linear solver with python PC
class MyLinearSolver(cashocs._utils.linalg.LinearSolver):
    def solve(self, fun, A, b, ksp_options=None, rtol=None, atol=None, P=None):
        self.comm = fun.function_space().mesh().mpi_comm()
        ksp = PETSc.KSP().create(self.comm)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(JacobiPC())

        if P is None:
            ksp.setOperators(A)
        else:
            ksp.setOperators(A, P)

        if fun is None:
            x, _ = A.getVecs()
        else:
            x = fun.vector().vec()

        if ksp_options is not None:
            cashocs._utils.linalg.setup_petsc_options([ksp], [ksp_options])

        ksp.setFromOptions()
        if rtol is not None:
            ksp.rtol = rtol
        if atol is not None:
            ksp.atol = atol

        ksp.solve(b, x)

        if ksp.getConvergedReason() < 0:
            raise Exception(
                f"PETSc KSP did not converge. Reason: {ksp.getConvergedReason()}"
            )

        if hasattr(PETSc, "garbage_cleanup"):
            ksp.destroy()
            PETSc.garbage_cleanup(comm=self.comm)

        if fun is not None:
            fun.vector().apply("")

        return x


ksp_options = {"ksp_type": "cg", "ksp_monitor_true_residual": None, "ksp_view": None}
my_linear_solver = MyLinearSolver()
cashocs.linear_solve(F, u, bcs, ksp_options=ksp_options, linear_solver=my_linear_solver)
