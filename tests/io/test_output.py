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

from collections import namedtuple

from fenics import *
import pytest

import cashocs


@pytest.fixture
def geometry():
    Geometry = namedtuple("Geometry", "mesh boundaries dx ds")
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(10)
    geom = Geometry(mesh, boundaries, dx, ds)

    return geom


@pytest.fixture
def CG1(geometry):
    return FunctionSpace(geometry.mesh, "CG", 1)


@pytest.fixture
def y(CG1):
    return Function(CG1)


@pytest.fixture
def p(CG1):
    return Function(CG1)


@pytest.fixture
def u(CG1):
    return Function(CG1)


@pytest.fixture
def F(y, u, p, geometry):
    return dot(grad(y), grad(p)) * geometry.dx - u * p * geometry.dx


@pytest.fixture
def bcs(CG1, geometry):
    return cashocs.create_dirichlet_bcs(
        CG1, Constant(0), geometry.boundaries, [1, 2, 3, 4]
    )


@pytest.fixture
def J(y, y_d, u, geometry):
    alpha = 1e-6
    return cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * geometry.dx
        + Constant(0.5 * alpha) * u * u * geometry.dx
    )


def test_time_suffix(config_ocp, work_dir, F, bcs, J, y, u, p):
    result_dir = work_dir / "results"
    config_ocp["Output"]["result_dir"] = str(result_dir)
    config_ocp["Output"]["time_suffix"] = True
    config_ocp["Output"]["save_txt"] = True
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    ocp.solve()
    suffix = ocp.solver.output_manager.suffix
    suffixed_result_dir = result_dir.with_name(f"results_{suffix}")
    assert suffixed_result_dir.is_dir()
    assert (suffixed_result_dir / "history.txt").is_file()
    MPI.barrier(MPI.comm_world)


def test_save_xdmf_files_ocp(work_dir, F, bcs, J, y, u, p, config_ocp):
    config_ocp["Output"]["save_state"] = True
    config_ocp["Output"]["save_results"] = True
    config_ocp["Output"]["save_txt"] = True
    config_ocp["Output"]["save_adjoint"] = True
    config_ocp["Output"]["save_gradient"] = True
    output_dir = work_dir / "out"
    config_ocp["Output"]["result_dir"] = str(output_dir)
    u.vector().vec().set(0.0)
    u.vector().apply("")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    ocp.solve(algorithm="bfgs", rtol=1e-1)
    MPI.barrier(MPI.comm_world)
    assert output_dir.is_dir()
    assert (output_dir / "checkpoints").is_dir()
    assert (output_dir / "history.txt").is_file()
    assert (output_dir / "history.json").is_file()
    assert (output_dir / "checkpoints/state_0.xdmf").is_file()
    assert (output_dir / "checkpoints/state_0.h5").is_file()
    assert (output_dir / "checkpoints/control_0.xdmf").is_file()
    assert (output_dir / "checkpoints/control_0.h5").is_file()
    assert (output_dir / "checkpoints/adjoint_0.xdmf").is_file()
    assert (output_dir / "checkpoints/adjoint_0.h5").is_file()
    assert (output_dir / "checkpoints/gradient_0.xdmf").is_file()
    assert (output_dir / "checkpoints/gradient_0.h5").is_file()

    MPI.barrier(MPI.comm_world)

    MPI.barrier(MPI.comm_world)


def test_save_xdmf_single_file_ocp(work_dir, F, bcs, J, y, u, p, config_ocp):
    config_ocp["Output"]["save_state"] = True
    config_ocp["Output"]["save_results"] = True
    config_ocp["Output"]["save_txt"] = True
    config_ocp["Output"]["save_adjoint"] = True
    config_ocp["Output"]["save_gradient"] = True
    config_ocp["Output"]["single_file"] = True
    output_dir = work_dir / "out"
    config_ocp["Output"]["result_dir"] = str(output_dir)
    u.vector().vec().set(0.0)
    u.vector().apply("")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    ocp.solve(algorithm="bfgs", rtol=1e-1)
    MPI.barrier(MPI.comm_world)
    checkpoints_dir = output_dir / "checkpoints"
    assert output_dir.is_dir()
    assert checkpoints_dir.is_dir()
    assert (output_dir / "history.txt").is_file()
    assert (output_dir / "history.json").is_file()
    assert (checkpoints_dir / "result.xdmf").is_file()
    assert (checkpoints_dir / "result.h5").is_file()
    assert not (checkpoints_dir / "state_0.xdmf").is_file()
    assert not (checkpoints_dir / "state_0.h5").is_file()
    assert not (checkpoints_dir / "control_0.xdmf").is_file()
    assert not (checkpoints_dir / "control_0.h5").is_file()
    assert not (checkpoints_dir / "adjoint_0.xdmf").is_file()
    assert not (checkpoints_dir / "adjoint_0.h5").is_file()
    assert not (checkpoints_dir / "gradient_0.xdmf").is_file()
    assert not (checkpoints_dir / "gradient_0.h5").is_file()

    MPI.barrier(MPI.comm_world)

    MPI.barrier(MPI.comm_world)


def test_save_xdmf_files_mixed(work_dir, rng, config_ocp, geometry):
    config_ocp["Output"]["save_state"] = True
    config_ocp["Output"]["save_results"] = True
    config_ocp["Output"]["save_txt"] = True
    config_ocp["Output"]["save_adjoint"] = True
    config_ocp["Output"]["save_gradient"] = True
    output_dir = work_dir / "out"
    config_ocp["Output"]["result_dir"] = str(output_dir)
    elem1 = VectorElement("CG", geometry.mesh.ufl_cell(), 2)
    elem2 = FiniteElement("CG", geometry.mesh.ufl_cell(), 1)
    V = FunctionSpace(geometry.mesh, MixedElement([elem1, elem2]))

    up = Function(V)
    u, p = split(up)
    vq = Function(V)
    v, q = split(vq)
    c = Function(V.sub(0).collapse())
    F = (
        inner(grad(u), grad(v)) * geometry.dx
        - p * div(v) * geometry.dx
        - q * div(u) * geometry.dx
        - inner(c, v) * geometry.dx
    )
    bcs = cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0.0, 0.0)), geometry.boundaries, [1, 2, 3, 4]
    )

    u_d = Expression(
        ("sin(2*pi*x[0])*sin(2*pi*x[1])", "sin(2*pi*x[0])*sin(2*pi*x[1])"),
        degree=1,
        domain=geometry.mesh,
    )
    J = cashocs.IntegralFunctional(
        Constant(0.5) * inner(u - u_d, u - u_d) * geometry.dx
    )

    ocp = cashocs.OptimalControlProblem(F, bcs, J, up, c, vq, config=config_ocp)
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9
    assert ocp.gradient_test(rng=rng) > 1.9
    ocp.solve(rtol=1e-1)

    MPI.barrier(MPI.comm_world)

    checkpoints_dir = output_dir / "checkpoints"
    assert output_dir.is_dir()
    assert checkpoints_dir.is_dir()
    assert (output_dir / "history.txt").is_file()
    assert (output_dir / "history.json").is_file()
    assert (checkpoints_dir / "state_0_0.xdmf").is_file()
    assert (checkpoints_dir / "state_0_0.h5").is_file()
    assert (checkpoints_dir / "state_0_1.xdmf").is_file()
    assert (checkpoints_dir / "state_0_1.h5").is_file()
    assert (checkpoints_dir / "control_0.xdmf").is_file()
    assert (checkpoints_dir / "control_0.h5").is_file()
    assert (checkpoints_dir / "adjoint_0_0.xdmf").is_file()
    assert (checkpoints_dir / "adjoint_0_0.h5").is_file()
    assert (checkpoints_dir / "adjoint_0_1.xdmf").is_file()
    assert (checkpoints_dir / "adjoint_0_1.h5").is_file()
    assert (checkpoints_dir / "gradient_0.xdmf").is_file()
    assert (checkpoints_dir / "gradient_0.h5").is_file()

    MPI.barrier(MPI.comm_world)

    MPI.barrier(MPI.comm_world)


@pytest.mark.slow
def test_extract_mesh_from_xdmf(work_dir, F, bcs, J, y, u, p, config_ocp):
    config_ocp["Output"]["save_state"] = True

    mesh_initial = u.function_space().mesh()
    result_path = work_dir / "out"
    config_ocp["Output"]["result_dir"] = str(result_path)
    u.vector().vec().set(0.0)
    u.vector().apply("")
    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config_ocp)
    ocp.solve(algorithm="bfgs", rtol=1e-1)
    MPI.barrier(MPI.comm_world)
    assert result_path.is_dir()

    cashocs.io.extract_mesh_from_xdmf(
        str(result_path / "checkpoints/state_0.xdmf"), iteration=3
    )
    assert (result_path / "checkpoints/state_0.msh").is_file()

    cashocs.io.extract_mesh_from_xdmf(
        str(result_path / "checkpoints/state_0.xdmf"),
        iteration=3,
        outputfile=str(result_path / "test.msh"),
    )
    assert (result_path / "test.msh").is_file()

    cashocs.convert(
        str(result_path / "test.msh"),
        output_file=str(result_path / "test.xdmf"),
        mode="geometrical",
    )
    mesh, _, _, _, _, _ = cashocs.import_mesh(str(result_path / "test.xdmf"))

    assert mesh.num_entities_global(0) == mesh_initial.num_entities_global(0)
    assert mesh.num_entities_global(2) == mesh_initial.num_entities_global(2)

    MPI.barrier(MPI.comm_world)

    MPI.barrier(MPI.comm_world)
