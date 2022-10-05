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

import os
import subprocess

from fenics import *
import mpi4py.MPI
import numpy as np

import cashocs

rng = np.random.RandomState(300696)
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
    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", "-r", f"{dir_path}/results_{suffix}"], check=True)
    MPI.barrier(MPI.comm_world)


def test_save_xdmf_files_ocp():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    config.set("Output", "save_state", "True")
    config.set("Output", "save_results", "True")
    config.set("Output", "save_txt", "True")
    config.set("Output", "save_adjoint", "True")
    config.set("Output", "save_gradient", "True")
    config.set("Output", "result_dir", dir_path + "/out")
    u.vector().vec().set(0.0)
    u.vector().apply("")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
    ocp.solve(algorithm="bfgs", rtol=1e-1)
    MPI.barrier(MPI.comm_world)
    assert os.path.isdir(dir_path + "/out")
    assert os.path.isdir(dir_path + "/out/xdmf")
    assert os.path.isfile(dir_path + "/out/history.txt")
    assert os.path.isfile(dir_path + "/out/history.json")
    assert os.path.isfile(dir_path + "/out/xdmf/state_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/state_0.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/control_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/control_0.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/adjoint_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/adjoint_0.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/gradient_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/gradient_0.h5")
    MPI.barrier(MPI.comm_world)

    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", "-r", f"{dir_path}/out"], check=True)
    MPI.barrier(MPI.comm_world)


def test_save_xdmf_files_mixed():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    config.set("Output", "save_state", "True")
    config.set("Output", "save_results", "True")
    config.set("Output", "save_txt", "True")
    config.set("Output", "save_adjoint", "True")
    config.set("Output", "save_gradient", "True")
    config.set("Output", "result_dir", dir_path + "/out")
    elem1 = VectorElement("CG", mesh.ufl_cell(), 2)
    elem2 = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement([elem1, elem2]))

    up = Function(V)
    u, p = split(up)
    vq = Function(V)
    v, q = split(vq)
    c = Function(V.sub(0).collapse())
    F = (
        inner(grad(u), grad(v)) * dx
        - p * div(v) * dx
        - q * div(u) * dx
        - inner(c, v) * dx
    )
    bcs = cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0.0, 0.0)), boundaries, [1, 2, 3, 4]
    )

    u_d = Expression(
        ("sin(2*pi*x[0])*sin(2*pi*x[1])", "sin(2*pi*x[0])*sin(2*pi*x[1])"),
        degree=1,
        domain=mesh,
    )
    J = Constant(0.5) * inner(u - u_d, u - u_d) * dx

    ocp = cashocs.OptimalControlProblem(F, bcs, J, up, c, vq, config)
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9
    ocp.solve(rtol=1e-1)

    MPI.barrier(MPI.comm_world)

    assert os.path.isdir(dir_path + "/out")
    assert os.path.isdir(dir_path + "/out/xdmf")
    assert os.path.isfile(dir_path + "/out/history.txt")
    assert os.path.isfile(dir_path + "/out/history.json")
    assert os.path.isfile(dir_path + "/out/xdmf/state_0_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/state_0_0.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/state_0_1.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/state_0_1.h5")

    assert os.path.isfile(dir_path + "/out/xdmf/control_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/control_0.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/adjoint_0_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/adjoint_0_0.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/adjoint_0_1.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/adjoint_0_1.h5")
    assert os.path.isfile(dir_path + "/out/xdmf/gradient_0.xdmf")
    assert os.path.isfile(dir_path + "/out/xdmf/gradient_0.h5")

    MPI.barrier(MPI.comm_world)

    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", "-r", f"{dir_path}/out"], check=True)
    MPI.barrier(MPI.comm_world)
