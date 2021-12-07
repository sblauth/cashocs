# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_inverse_tomography.html.

"""

from fenics import *

import cashocs



kappa_out = 1e0
kappa_in = 1e1


def generate_measurements():
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        "./mesh/reference.xdmf"
    )

    cg_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    r_elem = FiniteElement("R", mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, MixedElement([cg_elem, r_elem]))

    u, c = TrialFunctions(V)
    v, d = TestFunctions(V)

    a = (
        kappa_out * inner(grad(u), grad(v)) * dx(1)
        + kappa_in * inner(grad(u), grad(v)) * dx(2)
        + u * d * ds
        + v * c * ds
    )
    L1 = Constant(1) * v * (ds(3) + ds(4)) + Constant(-1) * v * (ds(1) + ds(2))
    L2 = Constant(1) * v * (ds(3) + ds(2)) + Constant(-1) * v * (ds(1) + ds(4))
    L3 = Constant(1) * v * (ds(3) + ds(1)) + Constant(-1) * v * (ds(2) + ds(4))

    meas1 = Function(V)
    meas2 = Function(V)
    meas3 = Function(V)
    solve(a == L1, meas1)
    solve(a == L2, meas2)
    solve(a == L3, meas3)

    m1, _ = meas1.split(True)
    m2, _ = meas2.split(True)
    m3, _ = meas3.split(True)

    return [m1, m2, m3]


config = cashocs.load_config("./config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh("./mesh/mesh.xdmf")
cg_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
r_elem = FiniteElement("R", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, MixedElement([cg_elem, r_elem]))

measurements = generate_measurements()

uc1 = Function(V)
u1, c1 = split(uc1)
pd1 = Function(V)
p1, d1 = split(pd1)
e1 = (
    kappa_out * inner(grad(u1), grad(p1)) * dx(1)
    + kappa_in * inner(grad(u1), grad(p1)) * dx(2)
    + u1 * d1 * ds
    + p1 * c1 * ds
    - Constant(1) * p1 * (ds(3) + ds(4))
    - Constant(-1) * p1 * (ds(1) + ds(2))
)

uc2 = Function(V)
u2, c2 = split(uc2)
pd2 = Function(V)
p2, d2 = split(pd2)
e2 = (
    kappa_out * inner(grad(u2), grad(p2)) * dx(1)
    + kappa_in * inner(grad(u2), grad(p2)) * dx(2)
    + u2 * d2 * ds
    + p2 * c2 * ds
    - Constant(1) * p2 * (ds(3) + ds(2))
    - Constant(-1) * p2 * (ds(1) + ds(4))
)

uc3 = Function(V)
u3, c3 = split(uc3)
pd3 = Function(V)
p3, d3 = split(pd3)
e3 = (
    kappa_out * inner(grad(u3), grad(p3)) * dx(1)
    + kappa_in * inner(grad(u3), grad(p3)) * dx(2)
    + u3 * d3 * ds
    + p3 * c3 * ds
    - Constant(1) * p3 * (ds(3) + ds(1))
    - Constant(-1) * p3 * (ds(2) + ds(4))
)

e = [e1, e2, e3]
u = [uc1, uc2, uc3]
p = [pd1, pd2, pd3]

bcs = None

J1 = Constant(0.5) * pow(u1 - measurements[0], 2) * ds
J2 = Constant(0.5) * pow(u2 - measurements[1], 2) * ds
J3 = Constant(0.5) * pow(u3 - measurements[2], 2) * ds

J = J1 + J2 + J3

sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
sop.solve()


### Post Processing

import matplotlib.pyplot as plt

DG0 = FunctionSpace(mesh, "DG", 0)
plt.figure(figsize=(10, 5))

result = Function(DG0)
a_post = TrialFunction(DG0) * TestFunction(DG0) * dx
L_post = Constant(1) * TestFunction(DG0) * dx(1) + Constant(2) * TestFunction(DG0) * dx(
    2
)
solve(a_post == L_post, result)

ax_result = plt.subplot(1, 2, 2)
fig_result = plot(result)
plt.title("Optimized Geometry")

mesh_initial, _, _, _, _, _ = cashocs.import_mesh("./mesh/mesh.xdmf")
mesh.coordinates()[:, :] = mesh_initial.coordinates()[:, :]
mesh.bounding_box_tree().build(mesh)
initial = Function(DG0)
solve(a_post == L_post, initial)

ax_initial = plt.subplot(1, 2, 1)
fig_initial = plot(initial)
plt.title("Initial Geometry")

plt.tight_layout()
# plt.savefig('./img_inverse_tomography.png', dpi=150, bbox_inches='tight')
