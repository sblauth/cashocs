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
import subprocess

import fenics
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs.io.mesh import gather_coordinates


def test_cli(dir_path):
    cashocs.convert(f"{dir_path}/mesh/mesh.msh", f"{dir_path}/mesh/mesh.xdmf")
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

    mesh_coords = gather_coordinates(mesh)
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert np.allclose(mesh_coords, gmsh_coords)
    fenics.MPI.barrier(fenics.MPI.comm_world)

    assert pathlib.Path(f"{dir_path}/mesh/mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_subdomains.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_subdomains.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.h5").is_file()

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def test_convert_wrapper(dir_path):
    cashocs.convert(f"{dir_path}/mesh/mesh.msh")
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

    mesh_coords = gather_coordinates(mesh)
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert np.allclose(mesh_coords, gmsh_coords)
    fenics.MPI.barrier(fenics.MPI.comm_world)

    assert pathlib.Path(f"{dir_path}/mesh/mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_subdomains.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_subdomains.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.h5").is_file()

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def test_convert3D(dir_path):
    cashocs.convert(f"{dir_path}/mesh/mesh3.msh", f"{dir_path}/mesh/mesh3.xdmf")
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

    assert pathlib.Path(f"{dir_path}/mesh/mesh3.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3_subdomains.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3_subdomains.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3_boundaries.h5").is_file()

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3_subdomains.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3_subdomains.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3_boundaries.h5"], check=True)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def test_wrong_formats(dir_path):
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        with pytest.raises(Exception) as e_info:
            cashocs._cli.convert(
                [f"{dir_path}/mesh/mesh.mesh", "-o", f"{dir_path}/mesh/mesh.xdmf"]
            )
        assert "due to wrong format." in str(e_info.value)
    fenics.MPI.barrier(fenics.MPI.comm_world)

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        with pytest.raises(Exception) as e_info:
            cashocs._cli.convert(
                [f"{dir_path}/mesh/mesh.msh", "-o", f"{dir_path}/mesh/mesh.test"]
            )
        assert "due to wrong format." in str(e_info.value)
    fenics.MPI.barrier(fenics.MPI.comm_world)
