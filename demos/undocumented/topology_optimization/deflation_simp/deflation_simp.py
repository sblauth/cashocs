# Copyright (C) 2020-2026 Fraunhofer ITWM, Sebastian Blauth and
# Leon Baeck
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

import cashocs

# load the config
config = cashocs.load_config("config.ini")

# import the mesh and geometry
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
    "./mesh/mesh_deflation.xdmf"
)

# define the permeability that depends on the density function
r = 0.25
alpha_1 = 2.5 / (100**2)
alpha_2 = 2.5 / (0.01**2)


def alpha(rho):
    return alpha_2 + (alpha_1 - alpha_2) * rho * (1 + r) / (rho + r)


# define the volume penalization and target volume
pen_vol = 10000.0
gamma = 0.5

# set up the function space
v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, v_elem * p_elem)
CG1 = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)

# define boundary conditions
inflow1 = Expression(("-144*(x[1]-1.0/6)*(x[1]-2.0/6)", "0.0"), degree=2)
inflow2 = Expression(("-144*(x[1]-4.0/6)*(x[1]-5.0/6)", "0.0"), degree=2)
bc_no_slip = cashocs.create_dirichlet_bcs(V.sub(0), Constant((0, 0)), boundaries, [1])
bc_inflow1 = cashocs.create_dirichlet_bcs(V.sub(0), inflow2, boundaries, [2])
bc_inflow2 = cashocs.create_dirichlet_bcs(V.sub(0), inflow1, boundaries, [3])
bc_outflow1 = cashocs.create_dirichlet_bcs(V.sub(0), inflow2, boundaries, [4])
bc_outflow2 = cashocs.create_dirichlet_bcs(V.sub(0), inflow1, boundaries, [5])
bcs = bc_no_slip + bc_inflow1 + bc_inflow2 + bc_outflow1 + bc_outflow2

# initialize the density function
rho = Function(CG1)
rho.vector()[:] = gamma

# set up state and adjoint variables
up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

# define the PDE constraint
F = (
    inner(grad(u), grad(v)) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    + alpha(rho) * dot(u, v) * dx
)

# define the control constraint for the density function
cc = [0.0, 1.0]

# set up the cost functional
J = cashocs.IntegralFunctional(
    Constant(0.5) * alpha(rho) * inner(u, u) * dx
    + Constant(0.5) * inner(grad(u), grad(u)) * dx
)
J_pen = cashocs.ScalarTrackingFunctional(rho * dx, gamma, pen_vol)

# define the deflated optimization problem and solve it
dtop = cashocs.DeflatedOptimalControlProblem(
    F, bcs, [J, J_pen], up, rho, vq, config=config, control_constraints=cc
)
dtop.solve(1e-6, 3, 0.4, 5000.0, 0.0, 0.0)

# import packages for the plotting
from matplotlib import colors
from matplotlib import pyplot as pp
import numpy as np

# define a costum color map
rgbvals = np.array([[0, 107, 164], [255, 128, 14]]) / 255.0
cmap = colors.LinearSegmentedColormap.from_list("tab10_colorblind", rgbvals, N=256)

# plot the computed density functions
for i in range(0, len(dtop.control_list_final)):
    plot(dtop.control_list_final[i], cmap=cmap, extend="max")
    pp.xticks([])
    pp.yticks([])
    pp.savefig("./results/shape_{i}.png".format(i=i), bbox_inches="tight")
