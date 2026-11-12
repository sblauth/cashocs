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


import subprocess

from mpi4py import MPI
import pytest

import cashocs


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
