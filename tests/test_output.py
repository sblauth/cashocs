# Copyright (C) 2020-2022 Sebastian Blauth
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

import os
import subprocess

from fenics import *

import cashocs


dir_path = os.path.dirname(os.path.realpath(__file__))
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx


def test_time_suffix():
    config.set("Output", "result_dir", f"{dir_path}/results")
    config.set("Output", "time_suffix", "True")
    config.set("Output", "save_txt", "True")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve()
    suffix = ocp.solver.output_manager.suffix
    assert os.path.isdir(dir_path + f"/results_{suffix}")
    assert os.path.isfile(dir_path + f"/results_{suffix}/history.txt")
    subprocess.run(["rm", "-r", f"{dir_path}/results_{suffix}"], check=True)
