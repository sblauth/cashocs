# Copyright (C) 2021 Sebastian Blauth
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

"""For the documentation of this demo see https://cashocs.readthedocs.io/en/latest/demos/optimal_control/doc_scalar_control_tracking.html.

"""

from fenics import *

import cashocs



cashocs.set_log_level(cashocs.LogLevel.INFO)
config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)
u.vector()[:] = 1.0

e = inner(grad(y), grad(p)) * dx - u * p * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

J = Constant(0) * dx

integrand = y * y * dx
tracking_goal = 1.0
J_tracking = {"integrand": integrand, "tracking_goal": tracking_goal}

ocp = cashocs.OptimalControlProblem(
    e, bcs, J, y, u, p, config, scalar_tracking_forms=J_tracking
)
ocp.solve()

print("L2-Norm of y squared: " + format(assemble(y * y * dx), ".3e"))


### Post Processing

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable u")

plt.subplot(1, 2, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable y")

plt.tight_layout()
# plt.savefig('./img_scalar_control_tracking.png', dpi=150, bbox_inches='tight')
