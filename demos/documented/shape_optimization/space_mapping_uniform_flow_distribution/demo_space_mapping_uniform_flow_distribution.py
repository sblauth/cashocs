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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_shape_poisson.html.

"""

import ctypes
import os
import subprocess

from fenics import *

import cashocs

space_mapping = cashocs.space_mapping.shape_optimization
cashocs.set_log_level(cashocs.LogLevel.ERROR)
dir = os.path.dirname(os.path.realpath(__file__))

Re = 100.0
cfg = cashocs.load_config("./config.ini")

# define the coarse model
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh("./mesh/mesh.xdmf")
n = FacetNormal(mesh)

u_in = Expression(("6.0*(0.0 - x[1])*(x[1] + 1.0)", "0.0"), degree=2)
q_in = -assemble(dot(u_in, n) * ds(1))
output_list = []

v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, v_elem * p_elem)

up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

F = inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx
bc_in = DirichletBC(V.sub(0), u_in, boundaries, 1)
bcs_wall = cashocs.create_dirichlet_bcs(
    V.sub(0), Constant((0.0, 0.0)), boundaries, [2, 3, 4]
)
bc_out = DirichletBC(V.sub(0).sub(0), Constant(0.0), boundaries, 5)
bc_pressure = DirichletBC(V.sub(1), Constant(0.0), boundaries, 5)
bcs = [bc_in] + bcs_wall + [bc_out] + [bc_pressure]

J = [cashocs.ScalarTrackingFunctional(dot(u, n) * ds(i), q_in / 3) for i in range(5, 8)]

coarse_model = space_mapping.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)

# define the fine model
class FineModel(space_mapping.FineModel):
    def __init__(self, mesh, Re, q_in, output_list):
        super().__init__(mesh)

        self.tracking_goals = [ctypes.c_double(0.0) for _ in range(5, 8)]

        self.iter = 0
        self.Re = Re
        self.q_in = q_in
        self.output_list = output_list

    def solve_and_evaluate(self):
        self.iter += 1

        # write out the mesh
        cashocs.io.write_out_mesh(
            self.mesh, "./mesh/mesh.msh", f"./mesh/fine/mesh_{self.iter}.msh"
        )
        cashocs.io.write_out_mesh(self.mesh, "./mesh/mesh.msh", f"./mesh/fine/mesh.msh")

        subprocess.run(
            ["gmsh", "./mesh/fine.geo", "-2", "-o", "./mesh/fine/fine.msh"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cashocs.convert("./mesh/fine/fine.msh", "./mesh/fine/fine.xdmf")

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
            "./mesh/fine/fine.xdmf"
        )
        n = FacetNormal(mesh)
        v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
        p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, v_elem * p_elem)

        up = Function(V)
        u, p = split(up)
        v, q = TestFunctions(V)

        F = (
            inner(grad(u), grad(v)) * dx
            + Constant(self.Re) * inner(grad(u) * u, v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )

        u_in = Expression(("6.0*(0.0 - x[1])*(x[1] + 1.0)", "0.0"), degree=2)
        bc_in = DirichletBC(V.sub(0), u_in, boundaries, 1)
        bcs_wall = cashocs.create_dirichlet_bcs(
            V.sub(0), Constant((0.0, 0.0)), boundaries, [2, 3, 4]
        )
        bc_out = DirichletBC(V.sub(0).sub(0), Constant(0.0), boundaries, 5)
        bc_pressure = DirichletBC(V.sub(1), Constant(0.0), boundaries, 5)
        bcs = [bc_in] + bcs_wall + [bc_out] + [bc_pressure]

        cashocs.newton_solve(F, up, bcs, verbose=False)
        self.u, p = up.split(True)

        file = File(f"./pvd/u_{self.iter}.pvd")
        file.write(self.u)

        J_list = [
            cashocs.ScalarTrackingFunctional(dot(self.u, n) * ds(i), self.q_in / 3)
            for i in range(5, 8)
        ]
        self.cost_functional_value = cashocs._utils.summation(
            [J.evaluate() for J in J_list]
        )

        self.flow_values = [assemble(dot(self.u, n) * ds(i)) for i in range(5, 8)]
        self.output_list.append(self.flow_values)

        for idx in range(len(self.tracking_goals)):
            self.tracking_goals[idx].value = self.flow_values[idx]


fine_model = FineModel(mesh, Re, q_in, output_list)

# define the parameter extraction step
up_param = Function(V)
u_param, p_param = split(up_param)
J_param = [
    cashocs.ScalarTrackingFunctional(
        dot(u_param, n) * ds(i), fine_model.tracking_goals[idx]
    )
    for idx, i in enumerate(range(5, 8))
]
parameter_extraction = space_mapping.ParameterExtraction(
    coarse_model, J_param, up_param, config=cfg
)


problem = space_mapping.SpaceMappingProblem(
    fine_model,
    coarse_model,
    parameter_extraction,
    method="broyden",
    max_iter=25,
    tol=1e-4,
    use_backtracking_line_search=False,
    broyden_type="good",
    memory_size=5,
    save_history=True,
)
problem.solve()


# Post-processing
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))

ax_coarse = plt.subplot(2, 2, 1)
fig_coarse = plot(mesh)
plt.title("Coarse Model Optimal Mesh")

ax_fine = plt.subplot(2, 2, 2)
fig_fine = plot(fine_model.u.function_space().mesh())
plt.title("Fine Model Optimal Mesh")

ax_coarse2 = plt.subplot(2, 2, 3)
fig_coarse2 = plot(u)
plt.title("Velocity (coarse model)")

ax_fine2 = plt.subplot(2, 2, 4)
fig_fine2 = plot(fine_model.u)
plt.title("Velocity (fine model)")

plt.tight_layout()
# plt.savefig(
#     "./img_space_mapping_uniform_flow_distribution.png", dpi=150, bbox_inches="tight"
# )
