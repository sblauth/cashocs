# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
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
from mpi4py import MPI
import numpy as np
import pytest

import cashocs
from cashocs.io.mesh import gather_coordinates


@pytest.mark.slow
@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test cannot be run in parallel.",
)
def test_convert_cli(copy_test_asset):
    mesh_file = copy_test_asset("data/meshes/unit_square/mesh.msh")
    xdmf_file = mesh_file.with_suffix(".xdmf")
    subprocess.run(
        ["cashocs-convert", str(mesh_file)],
        check=True,
    )
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(xdmf_file))

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

    assert xdmf_file.is_file()
    assert xdmf_file.with_suffix(".h5").is_file()
    assert xdmf_file.with_name("mesh_boundaries.xdmf").is_file()
    assert xdmf_file.with_name("mesh_boundaries.h5").is_file()


def test_convert_output_arg(mesh_dir, work_dir):
    output_file = work_dir / "test.xdmf"
    cashocs.convert(str(mesh_dir / "unit_square" / "mesh.msh"), str(output_file))
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(output_file))

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

    assert output_file.is_file()
    assert output_file.with_suffix(".h5").is_file()
    assert output_file.with_name("test_boundaries.xdmf").is_file()
    assert output_file.with_name("test_boundaries.h5").is_file()


def test_convert_wrapper(copy_test_asset):
    mesh_file = copy_test_asset("data/meshes/unit_square/mesh.msh")
    xdmf_file = mesh_file.with_suffix(".xdmf")
    cashocs.convert(str(mesh_file))
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(xdmf_file))

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

    assert xdmf_file.is_file()
    assert xdmf_file.with_suffix(".h5").is_file()
    assert xdmf_file.with_name("mesh_boundaries.xdmf").is_file()
    assert xdmf_file.with_name("mesh_boundaries.h5").is_file()


def test_convert3D(mesh_dir, work_dir):
    output_file = work_dir / "mesh3.xdmf"
    cashocs.convert(str(mesh_dir / "unit_cube" / "mesh3.msh"), str(output_file))
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(output_file))

    assert fenics.assemble(1 * dx) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds) == pytest.approx(6, rel=1e-14)

    assert fenics.assemble(1 * ds(1)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(4)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(5)) == pytest.approx(1, rel=1e-14)
    assert fenics.assemble(1 * ds(6)) == pytest.approx(1, rel=1e-14)

    assert output_file.is_file()
    assert output_file.with_suffix(".h5").is_file()
    assert output_file.with_name("mesh3_boundaries.xdmf").is_file()
    assert output_file.with_name("mesh3_boundaries.h5").is_file()


@pytest.mark.slow
def test_wrong_formats(mesh_dir, work_dir):
    subprocess_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_"))
    }
    result = None
    if MPI.COMM_WORLD.rank == 0:
        result = subprocess.run(
            [
                "cashocs-convert",
                str(mesh_dir / "unit_square" / "mesh.mesh"),
                "-o",
                str(work_dir / "mesh.xdmf"),
            ],
            capture_output=True,
            text=True,
            env=subprocess_env,
        )
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        assert result.returncode != 0
        assert "due to wrong format." in result.stderr

    result = None
    if MPI.COMM_WORLD.rank == 0:
        result = subprocess.run(
            [
                "cashocs-convert",
                str(mesh_dir / "unit_square" / "mesh.msh"),
                "-o",
                str(work_dir / "mesh.test"),
            ],
            capture_output=True,
            text=True,
            env=subprocess_env,
        )
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        assert result.returncode != 0
        assert "due to wrong format." in result.stderr


@pytest.mark.slow
@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test cannot be run in parallel.",
)
def test_extract_mesh_cli(xdmf_state_dir, work_dir):
    tmp_path = work_dir
    subprocess.run(
        [
            "cashocs-extract-mesh",
            str(xdmf_state_dir / "state_0.xdmf"),
            "-i",
            "3",
            "-o",
            str(tmp_path / "test.msh"),
        ],
        check=True,
    )
    assert (tmp_path / "test.msh").is_file()

    cashocs.convert(
        str(tmp_path / "test.msh"),
        output_file=str(tmp_path / "test.xdmf"),
        mode="geometrical",
    )
    mesh, _, _, _, _, _ = cashocs.import_mesh(str(tmp_path / "test.xdmf"))

    assert mesh.num_vertices() == 121
    assert mesh.num_cells() == 200

    MPI.COMM_WORLD.barrier()
