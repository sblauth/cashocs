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

import pathlib
import shutil
import subprocess
import sys
from unittest.mock import patch

from fenics import Constant
from fenics import DirichletBC
from fenics import Function
from fenics import FunctionSpace
from fenics import grad
from fenics import inner
from fenics import MPI
from fenics import SpatialCoordinate
import numpy as np
import pytest

import cashocs
from cashocs._exceptions import CashocsDebugException

rng = np.random.RandomState(300696)
has_gmsh = False
query = shutil.which("gmsh")
if query is not None:
    has_gmsh = True
else:
    has_gmsh = False

is_parallel = False
if MPI.comm_world.size > 1:
    is_parallel = True


dir_path = str(pathlib.Path(__file__).parent)

config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")

mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(config)

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
p = Function(V)

x = SpatialCoordinate(mesh)
f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

e = inner(grad(u), grad(p)) * dx - f * p * dx
bcs = DirichletBC(V, Constant(0), boundaries, 1)

J = cashocs.IntegralFunctional(u * dx)


def test_verification_remeshing():
    MPI.barrier(MPI.comm_world)
    dir_path = str(pathlib.Path(__file__).parent)

    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
    config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
    config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
    config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    MPI.barrier(MPI.comm_world)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    MPI.barrier(MPI.comm_world)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    MPI.barrier(MPI.comm_world)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    MPI.barrier(MPI.comm_world)


@pytest.mark.skipif(not has_gmsh, reason="This test requires Gmsh")
def test_first_remeshing_step():
    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
    config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
    config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
    config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")
    config.set("Output", "result_dir", dir_path + "/temp/")
    config.set("Debug", "remeshing", "True")
    config.set("Debug", "restart", "True")

    with patch.object(sys, "argv", [pathlib.Path(__file__).resolve()]):
        sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
        try:
            sop.solve(max_iter=10)
        except CashocsDebugException:
            pass

    MPI.barrier(MPI.comm_world)

    for s in pathlib.Path(f"{dir_path}/mesh/remesh").iterdir():
        print(f"DEBUG {s}")

    assert any(
        folder.name.startswith("cashocs_remesh_")
        for folder in pathlib.Path(f"{dir_path}/mesh/remesh").iterdir()
    )
    assert any(
        folder.name.startswith("._cashocs_remesh_temp_")
        for folder in pathlib.Path(f"{dir_path}").iterdir()
    )

    assert pathlib.Path(dir_path + "/temp").is_dir()
    assert pathlib.Path(dir_path + "/temp/xdmf").is_dir()
    assert pathlib.Path(dir_path + "/temp/history.txt").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/adjoint_0.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/adjoint_0.h5").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/state_0.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/state_0.h5").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/shape_gradient.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/shape_gradient.h5").is_file()

    MPI.barrier(MPI.comm_world)

    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", "-r", f"{dir_path}/temp"], check=True)

        subprocess.run(
            [f"rm -r {dir_path}/mesh/remesh/cashocs_remesh_*"], shell=True, check=True
        )
        subprocess.run(
            [f"rm -r {dir_path}/._cashocs_remesh_temp_*"], shell=True, check=True
        )
    MPI.barrier(MPI.comm_world)


def test_reentry():
    with patch.object(
        sys,
        "argv",
        [
            str(pathlib.Path(__file__).resolve()),
            "--cashocs_remesh",
            "--temp_dir",
            f"{dir_path}/temp_test_directory",
        ],
    ):
        config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
        config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
        config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
        config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")
        config.set("Output", "result_dir", dir_path + "/temp/")
        config.set("Debug", "remeshing", "True")

        MPI.barrier(MPI.comm_world)

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(config)

        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        p = Function(V)

        x = SpatialCoordinate(mesh)
        f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

        e = inner(grad(u), grad(p)) * dx - f * p * dx
        bcs = DirichletBC(V, Constant(0), boundaries, 1)

        J = cashocs.IntegralFunctional(u * dx)

        sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
        sop.solve(max_iter=10)

    MPI.barrier(MPI.comm_world)

    assert pathlib.Path(dir_path + "/temp").is_dir()
    assert pathlib.Path(dir_path + "/temp/xdmf").is_dir()
    assert pathlib.Path(dir_path + "/temp/history.txt").is_file()
    assert pathlib.Path(dir_path + "/temp/history.json").is_file()
    assert pathlib.Path(dir_path + "/temp/optimized_mesh.msh").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/adjoint_0.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/adjoint_0.h5").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/state_0.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/state_0.h5").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/shape_gradient.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/shape_gradient.h5").is_file()

    MPI.barrier(MPI.comm_world)

    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", "-r", f"{dir_path}/temp"], check=True)
    MPI.barrier(MPI.comm_world)


@pytest.mark.skipif(
    not has_gmsh or is_parallel,
    reason="This test requires Gmsh and cannot be run in parallel",
)
def test_remeshing():
    subprocess.run([sys.executable, f"{dir_path}/remeshing_script.py"], check=True)

    MPI.barrier(MPI.comm_world)
    assert any(
        folder.name.startswith("cashocs_remesh_")
        for folder in pathlib.Path(f"{dir_path}/mesh/remesh").iterdir()
    )
    assert any(
        folder.name.startswith("._cashocs_remesh_temp_")
        for folder in pathlib.Path(f"{dir_path}").iterdir()
    )

    assert pathlib.Path(dir_path + "/temp").is_dir()
    assert pathlib.Path(dir_path + "/temp/xdmf").is_dir()
    assert pathlib.Path(dir_path + "/temp/history.txt").is_file()
    assert pathlib.Path(dir_path + "/temp/history.json").is_file()
    assert pathlib.Path(dir_path + "/temp/optimized_mesh.msh").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/adjoint_0.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/adjoint_0.h5").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/state_0.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/state_0.h5").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/shape_gradient.xdmf").is_file()
    assert pathlib.Path(dir_path + "/temp/xdmf/shape_gradient.h5").is_file()

    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(
            [f"rm -r {dir_path}/mesh/remesh/cashocs_remesh_*"], shell=True, check=True
        )
        subprocess.run(
            [f"rm -r {dir_path}/._cashocs_remesh_temp_*"], shell=True, check=True
        )
        subprocess.run(["rm", "-r", f"{dir_path}/temp"], check=True)
    MPI.barrier(MPI.comm_world)


def test_remeshing_functionality():
    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
    config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
    config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
    config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    MPI.barrier(MPI.comm_world)
    assert pathlib.Path(f"{sop.mesh_handler.remesh_directory}/mesh_0.msh").is_file()

    sop.mesh_handler._generate_remesh_geo(config.get("Mesh", "gmsh_file"))
    assert pathlib.Path(f"{sop.mesh_handler.remesh_directory}/remesh.geo").is_file()

    if MPI.rank(MPI.comm_world) == 0:
        with open(f"{sop.mesh_handler.remesh_directory}/remesh.geo") as file:
            file_contents = file.read()
            test_contents = "Merge 'mesh.msh';\nCreateGeometry;\n\nlc = 5e-2;\nField[1] = Distance;\nField[1].NNodesByEdge = 1000;\nField[1].NodesList = {2};\nField[2] = Threshold;\nField[2].IField = 1;\nField[2].DistMin = 1e-1;\nField[2].DistMax = 5e-1;\nField[2].LcMin = lc / 10;\nField[2].LcMax = lc;\nBackground Field = 2;\n"

            assert file_contents == test_contents
    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", "-r", f"{sop.mesh_handler.remesh_directory}"], check=True)
    MPI.barrier(MPI.comm_world)


def test_remesh_scaling():
    rng = np.random.RandomState(300696)

    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
    config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
    config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
    config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")

    with patch.object(sys, "argv", [str(pathlib.Path(__file__).resolve())]):
        w_des = rng.rand(1)[0]
        sop = cashocs.ShapeOptimizationProblem(
            e, bcs, [J], u, p, boundaries, config, desired_weights=[w_des]
        )
        val = sop.reduced_cost_functional.evaluate()
        assert np.abs(np.abs(val) - w_des) < 1e-14

    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(
            [f"rm -r {dir_path}/mesh/remesh/cashocs_remesh_*"], shell=True, check=True
        )
        subprocess.run(
            [f"rm -r {dir_path}/._cashocs_remesh_temp_*"], shell=True, check=True
        )
    MPI.barrier(MPI.comm_world)
