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

import os

from fenics import *

import cashocs

space_mapping = cashocs.space_mapping.shape_optimization
cashocs.set_log_level(cashocs.LogLevel.ERROR)
dir = os.path.dirname(os.path.realpath(__file__))

# define model parameters and load config
alpha_1 = 1.0
alpha_2 = 10.0
f_1 = 1.0
f_2 = 10.0
beta = 100.0
cfg = cashocs.load_config("config.ini")


def create_desired_state(alpha_1, alpha_2, beta, f_1, f_2):
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        "./mesh/reference.xdmf"
    )
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = (
        Constant(alpha_1) * dot(grad(u), grad(v)) * dx(1)
        + Constant(alpha_2) * dot(grad(u), grad(v)) * dx(2)
        + Constant(beta) * pow(u, 3) * v * dx
        - Constant(f_1) * v * dx(1)
        - Constant(f_2) * v * dx(2)
    )
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])
    cashocs.newton_solve(F, u, bcs, verbose=False)

    return u


u_des_fixed = create_desired_state(alpha_1, alpha_2, beta, f_1, f_2)

# define the coarse model
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh("./mesh/mesh.xdmf")
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
p = Function(V)
u_des = Function(V)
F = (
    Constant(alpha_1) * dot(grad(u), grad(p)) * dx(1)
    + Constant(alpha_2) * dot(grad(u), grad(p)) * dx(2)
    - Constant(f_1) * p * dx(1)
    - Constant(f_2) * p * dx(2)
)
bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])
J = cashocs.IntegralFunctional(Constant(0.5) * pow(u - u_des, 2) * dx)
coarse_model = space_mapping.CoarseModel(F, bcs, J, u, p, boundaries, config=cfg)

# define the fine model
class FineModel(space_mapping.FineModel):
    def __init__(self, mesh, alpha_1, alpha_2, beta, f_1, f_2, u_des_fixed):
        super().__init__(mesh)
        self.u = Constant(0.0)
        self.iter = 0

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta = beta
        self.f_1 = f_1
        self.f_2 = f_2
        self.u_des_fixed = u_des_fixed

    def solve_and_evaluate(self) -> None:
        self.iter += 1
        cashocs.io.write_out_mesh(
            self.mesh, "./mesh/mesh.msh", f"./mesh/fine/mesh_{self.iter}.msh"
        )
        cashocs.convert(
            f"{dir}/mesh/fine/mesh_{self.iter}.msh", f"{dir}/mesh/fine/mesh.xdmf"
        )
        mesh, self.subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
            "./mesh/fine/mesh.xdmf"
        )

        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        u_des = Function(V)
        v = TestFunction(V)
        F = (
            Constant(self.alpha_1) * dot(grad(u), grad(v)) * dx(1)
            + Constant(self.alpha_2) * dot(grad(u), grad(v)) * dx(2)
            + Constant(self.beta) * pow(u, 3) * v * dx
            - Constant(self.f_1) * v * dx(1)
            - Constant(self.f_2) * v * dx(2)
        )
        bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])
        cashocs.newton_solve(F, u, bcs, verbose=False)

        LagrangeInterpolator.interpolate(u_des, self.u_des_fixed)

        self.cost_functional_value = assemble(Constant(0.5) * pow(u - u_des, 2) * dx)
        self.u = u


fine_model = FineModel(mesh, alpha_1, alpha_2, beta, f_1, f_2, u_des_fixed)
u_fine = Function(V)


def callback():
    LagrangeInterpolator.interpolate(u_des, u_des_fixed)
    LagrangeInterpolator.interpolate(u_fine, fine_model.u)


# define the parameter extraction step
u_param = Function(V)
J_param = cashocs.IntegralFunctional(Constant(0.5) * pow(u_param - u_fine, 2) * dx)
parameter_extraction = space_mapping.ParameterExtraction(
    coarse_model, J_param, u_param, config=cfg, mode="initial"
)


problem = space_mapping.SpaceMapping(
    fine_model,
    coarse_model,
    parameter_extraction,
    method="broyden",
    max_iter=25,
    tol=1e-2,
    use_backtracking_line_search=False,
    broyden_type="good",
    memory_size=5,
    verbose=True,
    save_history=True,
)
problem.inject_pre_callback(callback)
problem.solve()

# Post-processing
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

ax_coarse = plt.subplot(1, 3, 1)
fig_coarse = plot(subdomains)
plt.title("Coarse Model Optimal Geometry")

ax_fine = plt.subplot(1, 3, 2)
fig_fine = plot(fine_model.subdomains)
plt.title("Fine Model Optimal Geometry")

mesh, subdomains, _, _, _, _ = cashocs.import_mesh("./mesh/reference.xdmf")
ax_ref = plt.subplot(1, 3, 3)
fig_ref = plot(subdomains)
plt.title("Reference Geometry")

plt.tight_layout()
plt.savefig(
    "./img_space_mapping_semilinear_transmission.png", dpi=150, bbox_inches="tight"
)
