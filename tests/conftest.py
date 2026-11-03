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

import hashlib
import pathlib
import shutil
import tempfile

import fenics
from mpi4py import MPI
import numpy as np
import pytest

import cashocs


@pytest.fixture(scope="session")
def tests_dir():
    return pathlib.Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def data_dir(tests_dir):
    return pathlib.Path(tests_dir / "data")


@pytest.fixture(scope="session")
def mesh_dir(data_dir):
    return data_dir / "meshes"


@pytest.fixture(scope="session")
def sm_mesh_dir(mesh_dir):
    return mesh_dir / "sm_mesh"


@pytest.fixture(scope="session")
def xdmf_state_dir(data_dir):
    return data_dir / "xdmf_state"


@pytest.fixture(scope="session")
def config_dir(data_dir):
    return data_dir / "config"


@pytest.fixture(scope="session")
def shared_work_root(tests_dir):
    path = None
    error = None
    if MPI.COMM_WORLD.rank == 0:
        try:
            path = pathlib.Path(
                tempfile.mkdtemp(prefix=".cashocs-test-work-", dir=tests_dir)
            )
        except OSError as exception:
            error = f"{type(exception).__name__}: {exception}"
    path, error = MPI.COMM_WORLD.bcast(
        (str(path) if path is not None else None, error), root=0
    )
    if error is not None:
        raise RuntimeError(f"Unable to create shared test workspace: {error}")

    yield pathlib.Path(path)

    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(path)


@pytest.fixture
def work_dir(shared_work_root, request):
    path = shared_work_root / hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    error = None
    if MPI.COMM_WORLD.rank == 0:
        try:
            path.mkdir()
        except OSError as exception:
            error = f"{type(exception).__name__}: {exception}"
    path, error = MPI.COMM_WORLD.bcast(
        (str(path) if error is None else None, error), root=0
    )
    if error is not None:
        raise RuntimeError(f"Unable to create test workspace: {error}")

    yield pathlib.Path(path)

    MPI.COMM_WORLD.barrier()
    error = None
    if MPI.COMM_WORLD.rank == 0:
        try:
            shutil.rmtree(path)
        except OSError as exception:
            error = f"{type(exception).__name__}: {exception}"
    error = MPI.COMM_WORLD.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"Unable to remove test workspace: {error}")


@pytest.fixture
def copy_test_asset(tests_dir, work_dir):
    def copy_asset(relative_path):
        source = tests_dir / relative_path
        destination = work_dir / relative_path
        error = None

        if MPI.COMM_WORLD.rank == 0:
            try:
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            except OSError as exception:
                error = f"{type(exception).__name__}: {exception}"
        error = MPI.COMM_WORLD.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"Unable to copy test asset {source}: {error}")

        return destination

    return copy_asset


@pytest.fixture
def config_ocp(config_dir):
    return cashocs.load_config(str(config_dir / "config_ocp.ini"))


@pytest.fixture
def config_sop(config_dir):
    return cashocs.load_config(str(config_dir / "config_sop.ini"))


@pytest.fixture
def config_top(config_dir):
    return cashocs.load_config(str(config_dir / "config_top.ini"))


@pytest.fixture
def rng():
    return np.random.RandomState(300696)


@pytest.fixture
def y_d(geometry):
    return fenics.Expression(
        "sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=geometry.mesh
    )


@pytest.fixture(params=["FR", "PR", "HS", "DY", "HZ"])
def setup_cg_method(config_ocp, request):
    config_ocp["AlgoCG"]["cg_method"] = request.param


@pytest.fixture(params=["cr", "cg"])
def setup_newton_method(config_ocp, request):
    config_ocp["AlgoTNM"]["inner_newton"] = request.param
