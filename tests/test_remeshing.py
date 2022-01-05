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
import shutil
import subprocess

import numpy as np
import pytest
from fenics import *

import cashocs
import cashocs.config


rng = np.random.RandomState(300696)
has_gmsh = False
query = shutil.which("gmsh")
if query is not None:
    has_gmsh = True
else:
    has_gmsh = False


dir_path = os.path.dirname(os.path.realpath(__file__))


def test_verification_remeshing():
    dir_path = os.path.dirname(os.path.realpath(__file__))

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

    J = u * dx

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9


@pytest.mark.skipif(not has_gmsh, reason="This test requires Gmsh")
def test_remeshing():
    subprocess.run(["python", f"{dir_path}/remeshing_script.py"], check=True)

    assert any(
        s.startswith("cashocs_remesh_") for s in os.listdir(f"{dir_path}/mesh/remesh")
    )
    assert any(
        s.startswith("._cashocs_remesh_temp_") for s in os.listdir(f"{dir_path}")
    )

    assert os.path.isdir(dir_path + "/temp")
    assert os.path.isdir(dir_path + "/temp/pvd")
    assert os.path.isfile(dir_path + "/temp/history.txt")
    assert os.path.isfile(dir_path + "/temp/history.json")
    assert os.path.isfile(dir_path + "/temp/optimized_mesh.msh")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_adjoint_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_adjoint_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_adjoint_0000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_adjoint_0000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_state_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_state_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_state_0000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_state_0000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_shape_gradient.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_shape_gradient.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_shape_gradient000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_shape_gradient000003.vtu")

    subprocess.run(
        [f"rm -r {dir_path}/mesh/remesh/cashocs_remesh_*"], shell=True, check=True
    )
    subprocess.run(
        [f"rm -r {dir_path}/._cashocs_remesh_temp_*"], shell=True, check=True
    )
    subprocess.run(["rm", "-r", f"{dir_path}/temp"], check=True)


def test_remeshing_functionality():
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

    J = u * dx

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    assert os.path.isfile(f"{sop.mesh_handler.remesh_directory}/mesh_0.msh")

    sop.mesh_handler._MeshHandler__generate_remesh_geo(config.get("Mesh", "gmsh_file"))
    assert os.path.isfile(f"{sop.mesh_handler.remesh_directory}/remesh.geo")

    with open(f"{sop.mesh_handler.remesh_directory}/remesh.geo") as file:
        file_contents = file.read()
        test_contents = "Merge 'mesh.msh';\nCreateGeometry;\n\nlc = 5e-2;\nField[1] = Distance;\nField[1].NNodesByEdge = 1000;\nField[1].NodesList = {2};\nField[2] = Threshold;\nField[2].IField = 1;\nField[2].DistMin = 1e-1;\nField[2].DistMax = 5e-1;\nField[2].LcMin = lc / 10;\nField[2].LcMax = lc;\nBackground Field = 2;\n"

        assert file_contents == test_contents

    subprocess.run(["rm", "-r", f"{sop.mesh_handler.remesh_directory}"], check=True)
