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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/optimal_control/doc_pre_post_hooks.html.

"""

from fenics import *

import cashocs

config = cashocs.load_config("./config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(30)

v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))
U = VectorFunctionSpace(mesh, "CG", 1)

up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)
c = Function(U)

Re = Constant(1e2)

e = (
    inner(grad(u), grad(v)) * dx
    + Constant(Re) * dot(grad(u) * u, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    - inner(c, v) * dx
)


def pressure_point(x, on_boundary):
    return near(x[0], 0) and near(x[1], 0)


no_slip_bcs = cashocs.create_dirichlet_bcs(
    V.sub(0), Constant((0, 0)), boundaries, [1, 2, 3]
)
lid_velocity = Expression(("4*x[0]*(1-x[0])", "0.0"), degree=2)
bc_lid = DirichletBC(V.sub(0), lid_velocity, boundaries, 4)
bc_pressure = DirichletBC(V.sub(1), Constant(0), pressure_point, method="pointwise")
bcs = no_slip_bcs + [bc_lid, bc_pressure]

alpha = 1e-5
u_d = Expression(
    (
        "sqrt(pow(x[0], 2) + pow(x[1], 2))*cos(2*pi*x[1])",
        "-sqrt(pow(x[0], 2) + pow(x[1], 2))*sin(2*pi*x[0])",
    ),
    degree=2,
)
J = cashocs.IntegralFunctional(
    Constant(0.5) * inner(u - u_d, u - u_d) * dx
    + Constant(0.5 * alpha) * inner(c, c) * dx
)


def pre_hook():
    print("Solving low Re Navier-Stokes equations for homotopy.")
    v, q = TestFunctions(V)
    e = (
        inner(grad(u), grad(v)) * dx
        + Constant(Re / 10.0) * dot(grad(u) * u, v) * dx
        - p * div(v) * dx
        - q * div(u) * dx
        - inner(c, v) * dx
    )
    cashocs.newton_solve(e, up, bcs, verbose=False)


def post_hook():
    print("Performing an action after computing the gradient.")


ocp = cashocs.OptimalControlProblem(e, bcs, J, up, c, vq, config)
ocp.inject_pre_hook(pre_hook)
ocp.inject_post_hook(post_hook)
ocp.solve()


### Post Processing

u, p = up.split(True)
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
fig = plot(c)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable c")

plt.subplot(1, 3, 2)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable u")

plt.subplot(1, 3, 3)
fig = plot(u_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Desired state u_d")

plt.tight_layout()
# plt.savefig("./img_pre_post_hooks.png", dpi=150, bbox_inches="tight")
