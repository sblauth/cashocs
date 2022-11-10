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

uses_ompi = False
query = shutil.which("ompi_info")
if query is not None:
    uses_ompi = True


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


@pytest.mark.skipif(
    not has_gmsh or is_parallel or uses_ompi,
    reason="This test requires Gmsh and cannot be run in parallel. "
    "Remeshing in parallel only works for mpich, not openmpi",
)
def test_remeshing():
    subprocess.run(
        ["mpirun", "-n", "2", sys.executable, f"{dir_path}/remeshing_script.py"],
        check=True,
    )

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
