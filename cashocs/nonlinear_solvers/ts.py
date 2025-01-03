from petsc4py import PETSc
from cashocs import _utils
from cashocs import log
import fenics


class TSSolver:
    def __init__(
        self,
        nonlinear_form,
        u,
        bcs,
        petsc_options=None,
        derivative=None,
        preconditioner_form=None,
    ) -> None:
        self.nonlinear_form = nonlinear_form
        self.u = u

        self.space = self.u.function_space()
        self.mesh = self.space.mesh()
        self.comm = self.mesh.mpi_comm()
        self.bcs = _utils.enlist(bcs)

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

        self.A_petsc = fenics.PETScMatrix(self.comm).mat()

        self.mass_matrix_petsc = fenics.PETScMatrix(self.comm).mat()

        self.mass_application_petsc = fenics.Function(self.space).vector().vec()
        self.residual_fenics = fenics.PETScVector(self.comm)
        self.residual_petsc = self.residual_fenics.vec()

        self.assembler = fenics.SystemAssembler(
            self.derivative, self.nonlinear_form, self.bcs
        )
        self.assembler.keep_diagonal = True

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

        self.is_preassembled = False
        self.container = fenics.Function(self.space)
        self._setup_mass_matrix()

    def _setup_mass_matrix(self):
        space = self.u.function_space()
        trial = fenics.TrialFunction(space)
        test = fenics.TestFunction(space)

        dx = fenics.Measure("dx", domain=self.mesh)

        trial_u, _ = fenics.split(trial)
        test_u, _ = fenics.split(test)

        form = fenics.dot(trial_u, test_u) * dx
        M = fenics.PETScMatrix(self.mass_matrix_petsc)
        fenics.assemble(form, tensor=M)

        self.M_petsc = self.mass_matrix_petsc.copy()

        if self.preconditioner_form is not None:
            self.MP_petsc = self.mass_matrix_petsc.copy()
        else:
            self.MP_petsc = None

    def assemble_residual(self, ts, t, u, f):
        log.begin("Assembling the residual for pseudo time stepping.", level=log.DEBUG)
        self.u.vector().vec().setArray(u)
        self.u.vector().apply("")

        f_fenics = fenics.PETScVector(f)
        self.assembler.assemble(f_fenics, self.u.vector())

        f.scale(-1)
        log.end()

    def assemble_jacobian(self, ts, t, u, J, P):
        log.begin("Assembling the Jacobian for pseudo time stepping.", level=log.DEBUG)
        self.u.vector().vec().setArray(u)
        self.u.vector().apply("")

        J = fenics.PETScMatrix(J)  # pylint: disable=invalid-name
        self.assembler.assemble(J)
        J.ident_zeros()

        J_petsc = fenics.as_backend_type(J).mat()
        J_petsc.scale(-1)
        J_petsc.assemble()

        if self.preconditioner_form is not None:
            P = fenics.PETScMatrix(P)  # pylint: disable=invalid-name
            self.assembler.assemble(P)
            P.ident_zeros()

            P_petsc = fenics.as_backend_type(P).mat()
            P_petsc.scale(-1)
            P_petsc.assemble()

        log.end()

    def assemble_i_function(self, ts, t, u, u_dot, f):
        res = self.mass_matrix_petsc.createVecLeft()
        self.mass_matrix_petsc.mult(u_dot, res)
        f.setArray(res)

    def assemble_mass_matrix(self, ts, t, u, u_dot, sigma, A, B):
        A.aypx(0.0, self.mass_matrix_petsc)
        A.scale(sigma)
        A.assemble()

        if self.preconditioner_form is not None:
            B.aypx(0.0, self.mass_matrix_petsc)
            B.scale(sigma)
            B.assemble()

    def monitor(self, ts, i, t, x):
        # self.container.vector().vec().aypx(0.0, x)
        # append = False if i == 0 else True
        # with fenics.XDMFFile(self.mesh.mpi_comm(), "progress.xdmf") as file:
        #     file.parameters["flush_output"] = True
        #     file.parameters["functions_share_mesh"] = False
        #     file.write_checkpoint(
        #         self.container,
        #         "progress",
        #         i,
        #         fenics.XDMFFile.Encoding.HDF5,
        #         append,
        #     )

        residual_norm = self.residual_fenics.norm("l2")
        log.info(f"{i = }  {t = :.3e}  residual: {residual_norm:.3e}")

        if residual_norm < 1e-5 * self.res_initial:
            ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)
            max_time = ts.getMaxTime()
            ts.setTime(max_time)

    def solve(self):
        log.begin("Solving the nonlinear PDE system with pseudo time stepping.")
        ts = PETSc.TS().create()
        ts.setProblemType(ts.ProblemType.NONLINEAR)

        ts.setIFunction(self.assemble_i_function, self.mass_application_petsc)
        ts.setIJacobian(self.assemble_mass_matrix, self.M_petsc, self.MP_petsc)
        ts.setRHSFunction(self.assemble_residual, self.residual_petsc)
        ts.setRHSJacobian(self.assemble_jacobian, self.A_petsc, self.P_petsc)

        ksp = ts.getSNES().getKSP()
        _utils.linalg.setup_fieldsplit_preconditioner(self.u, ksp, self.petsc_options)

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
        self.res_initial = self.residual_fenics.norm("l2")

        ts.setTime(0.0)
        # ts.setTimeStep(1e-6)
        ts.setMaxTime(1e12)

        ts.setMonitor(self.monitor)

        _utils.setup_petsc_options([ts], [self.petsc_options])
        ts.solve(self.u.vector().vec())

        if hasattr(PETSc, "garbage_cleanup"):
            ts.destroy()
            PETSc.garbage_cleanup(comm=self.comm)
            PETSc.garbage_cleanup()

        log.end()

        return self.u


def ts_solve(
    nonlinear_form,
    u,
    bcs,
    petsc_options=None,
    derivative=None,
    preconditioner_form=None,
):
    solver = TSSolver(
        nonlinear_form,
        u,
        bcs,
        petsc_options=petsc_options,
        derivative=derivative,
        preconditioner_form=preconditioner_form,
    )

    solution = solver.solve()
    return solution
