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

import fenics
import numpy as np
import pytest

import cashocs
import cashocs._cli



dir_path = os.path.dirname(os.path.realpath(__file__))


def test_cli():
    cashocs._cli.convert([f"{dir_path}/mesh/mesh.msh", f"{dir_path}/mesh/mesh.xdmf"])
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/mesh.xdmf"
    )

    gmsh_coords = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.499999999998694, 0],
            [1, 0.499999999998694],
            [0.5000000000020591, 1],
            [0, 0.5000000000020591],
            [0.2500000000010297, 0.7500000000010296],
            [0.3749999970924328, 0.3750000029075671],
            [0.7187499979760099, 0.2812500030636815],
            [0.6542968741702071, 0.6542968818888233],
        ]
    )

    assert abs(fenics.assemble(1 * dx) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds) - 4) < 1e-14

    assert abs(fenics.assemble(1 * ds(1)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(2)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(3)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(4)) - 1) < 1e-14

    assert np.allclose(mesh.coordinates(), gmsh_coords)

    assert os.path.isfile(f"{dir_path}/mesh/mesh.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh.h5")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_subdomains.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_subdomains.h5")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_boundaries.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_boundaries.h5")

    subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)


def test_convert3D():
    cashocs._cli.convert([f"{dir_path}/mesh/mesh3.msh", f"{dir_path}/mesh/mesh3.xdmf"])
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/mesh3.xdmf"
    )

    assert abs(fenics.assemble(1 * dx) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds) - 6) < 1e-14

    assert abs(fenics.assemble(1 * ds(1)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(2)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(3)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(4)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(5)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(6)) - 1) < 1e-14

    assert os.path.isfile(f"{dir_path}/mesh/mesh3.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh3.h5")
    assert os.path.isfile(f"{dir_path}/mesh/mesh3_subdomains.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh3_subdomains.h5")
    assert os.path.isfile(f"{dir_path}/mesh/mesh3_boundaries.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh3_boundaries.h5")

    subprocess.run(["rm", f"{dir_path}/mesh/mesh3.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh3.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh3_subdomains.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh3_subdomains.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh3_boundaries.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh3_boundaries.h5"], check=True)


def test_wrong_formats():
    with pytest.raises(Exception) as e_info:
        cashocs._cli.convert(
            [f"{dir_path}/mesh/mesh.mesh", f"{dir_path}/mesh/mesh.xdmf"]
        )
    assert "Cannot use the input file due to wrong format." in str(e_info.value)

    with pytest.raises(Exception) as e_info:
        cashocs._cli.convert(
            [f"{dir_path}/mesh/mesh.msh", f"{dir_path}/mesh/mesh.test"]
        )
    assert "Cannot use the output file due to wrong format." in str(e_info.value)
