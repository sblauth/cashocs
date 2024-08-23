from fenics import *

import cashocs

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(16)
v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement(v_elem, p_elem, p_elem))
# order of the MixedElement determines (single) prefixes for PETSc solver
# velocity block gets 0, pressure 1, and temperature 2

upT = Function(V)
u, p, T = split(upT)
v, q, S = TestFunctions(V)

mu = 1.0 / (T + 1)

F = (
    mu * inner(grad(u), grad(v)) * dx
    + dot(grad(u) * u, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    + dot(grad(T), grad(S)) * dx
    + dot(u, grad(T)) * S * dx
    - Constant(100.0) * S * dx
)
u_in = Expression(("4.0 * x[1] * (1.0 - x[1])", "0.0"), degree=2)
bcs = cashocs.create_dirichlet_bcs(V.sub(0), u_in, boundaries, 1)
bcs += cashocs.create_dirichlet_bcs(V.sub(0), Constant((0.0, 0.0)), boundaries, [3, 4])
bcs += cashocs.create_dirichlet_bcs(V.sub(2), Constant(1.0), boundaries, [1, 3, 4])

petsc_options = {
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_max_it": 7,
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-1,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0,1",  # will get prefix fieldsplit_0_
    "pc_fieldsplit_1_fields": "2",  # will get prefix fieldsplit_2_
    "fieldsplit_0_ksp_type": "fgmres",
    "fieldsplit_0_ksp_rtol": 1e-1,
    "fieldsplit_0_pc_type": "fieldsplit",  # sub fields get global (mixed) index
    "fieldsplit_0_pc_fieldsplit_type": "schur",
    "fieldsplit_0_pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_0_fieldsplit_1_ksp_rtol": 1e-1,
    "fieldsplit_0_fieldsplit_1_pc_type": "hypre",
    "fieldsplit_0_fieldsplit_1_ksp_converged_reason": None,
    "fieldsplit_2_ksp_type": "gmres",
    "fieldsplit_2_ksp_rtol": 1e-1,
    "fieldsplit_2_pc_type": "hypre",
}

cashocs.snes_solve(F, upT, bcs, petsc_options=petsc_options, max_iter=8)
u, p, T = upT.split(True)
