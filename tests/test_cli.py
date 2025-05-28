# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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
from mpi4py import MPI
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs.io.mesh import gather_coordinates


@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test cannot be run in parallel.",
)
def test_convert_cli(dir_path):
    subprocess.run(
        ["cashocs-convert", f"{dir_path}/mesh/mesh.msh"],
        check=True,
    )
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

    assert fenics.assemble(1 * dx) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds) == pytest.approx(4.0, rel=1e-14)

    assert fenics.assemble(1 * ds(1)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(4)) == pytest.approx(1.0, rel=1e-14)

    mesh_coords = gather_coordinates(mesh)
    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(mesh_coords, gmsh_coords)
    MPI.COMM_WORLD.barrier()

    assert pathlib.Path(f"{dir_path}/mesh/mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.h5").is_file()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
    MPI.COMM_WORLD.barrier()


def test_convert_output_arg(dir_path):
    cashocs.convert(f"{dir_path}/mesh/mesh.msh", f"{dir_path}/mesh/test.xdmf")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/test.xdmf"
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

    assert fenics.assemble(1 * dx) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds) == pytest.approx(4.0, rel=1e-14)

    assert fenics.assemble(1 * ds(1)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(4)) == pytest.approx(1.0, rel=1e-14)

    mesh_coords = gather_coordinates(mesh)
    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(mesh_coords, gmsh_coords)
    MPI.COMM_WORLD.barrier()

    assert pathlib.Path(f"{dir_path}/mesh/test.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/test.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/test_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/test_boundaries.h5").is_file()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/test.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test_boundaries.h5"], check=True)
    MPI.COMM_WORLD.barrier()


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

    assert fenics.assemble(1 * dx) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds) == pytest.approx(4.0, rel=1e-14)

    assert fenics.assemble(1 * ds(1)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(4)) == pytest.approx(1.0, rel=1e-14)

    mesh_coords = gather_coordinates(mesh)
    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(mesh_coords, gmsh_coords)
    MPI.COMM_WORLD.barrier()

    assert pathlib.Path(f"{dir_path}/mesh/mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.h5").is_file()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
    MPI.COMM_WORLD.barrier()


def test_convert3D(dir_path):
    cashocs.convert(f"{dir_path}/mesh/mesh3.msh", f"{dir_path}/mesh/mesh3.xdmf")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/mesh3.xdmf"
    )

    assert fenics.assemble(1 * dx) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds) == pytest.approx(6, rel=1e-14)

    assert fenics.assemble(1 * ds(1)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(4)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(5)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(6)) == pytest.approx(1, rel=1e-14)

    assert pathlib.Path(f"{dir_path}/mesh/mesh3.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh3_boundaries.h5").is_file()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh3_boundaries.h5"], check=True)
    MPI.COMM_WORLD.barrier()


def test_wrong_formats(dir_path):
    if MPI.COMM_WORLD.rank == 0:
        with pytest.raises(Exception) as e_info:
            cashocs._cli.convert(
                [f"{dir_path}/mesh/mesh.mesh", "-o", f"{dir_path}/mesh/mesh.xdmf"]
            )
        assert "due to wrong format." in str(e_info.value)
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        with pytest.raises(Exception) as e_info:
            cashocs._cli.convert(
                [f"{dir_path}/mesh/mesh.msh", "-o", f"{dir_path}/mesh/mesh.test"]
            )
        assert "due to wrong format." in str(e_info.value)
    MPI.COMM_WORLD.barrier()


@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test cannot be run in parallel.",
)
def test_extract_mesh_cli(dir_path):
    tmp_path = pathlib.Path(f"{dir_path}/tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "cashocs-extract_mesh",
            f"{dir_path}/xdmf_state/state_0.xdmf",
            "-i",
            "3",
            "-o",
            f"{dir_path}/tmp/test.msh",
        ],
        check=True,
    )
    assert pathlib.Path(f"{dir_path}/tmp/test.msh").is_file()

    cashocs.convert(
        f"{dir_path}/tmp/test.msh",
        output_file=f"{dir_path}/tmp/test.xdmf",
        mode="geometrical",
    )
    mesh, _, _, _, _, _ = cashocs.import_mesh(f"{dir_path}/tmp/test.xdmf")

    assert mesh.num_vertices() == 121
    assert mesh.num_cells() == 200

    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", "-r", f"{dir_path}/tmp"], check=True)
    MPI.COMM_WORLD.barrier()
