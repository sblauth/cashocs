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

import shutil

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


def test_verification_remeshing(copy_test_asset, config_dir):
    MPI.barrier(MPI.comm_world)
    remesh_dir = copy_test_asset("data/meshes/remesh")
    mesh_file = remesh_dir / "mesh.xdmf"

    def mesh_parametrization(mesh_file):
        config = cashocs.load_config(str(config_dir / "config_remesh.ini"))
        config["Mesh"]["gmsh_file"] = str(remesh_dir / "mesh.msh")
        config["Mesh"]["geo_file"] = str(remesh_dir / "mesh.geo")

        config["Output"]["save_results"] = False
        config["Output"]["save_txt"] = False
        config["Output"]["save_state"] = False
        config["Output"]["save_adjoint"] = False
        config["Output"]["save_gradient"] = False
        config["Output"]["save_mesh"] = False

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(mesh_file)

        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        p = Function(V)

        x = SpatialCoordinate(mesh)
        f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

        e = inner(grad(u), grad(p)) * dx - f * p * dx
        bcs = DirichletBC(V, Constant(0), boundaries, 1)

        J = cashocs.IntegralFunctional(u * dx)

        args = (e, bcs, J, u, p, boundaries, config)
        kwargs = {}

        return args, kwargs

    sop = cashocs.ShapeOptimizationProblem(mesh_parametrization, str(mesh_file))
    MPI.barrier(MPI.comm_world)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    MPI.barrier(MPI.comm_world)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    MPI.barrier(MPI.comm_world)
    assert cashocs.verification.shape_gradient_test(sop, rng=rng) > 1.9
    MPI.barrier(MPI.comm_world)


@pytest.mark.slow
@pytest.mark.skipif(
    not has_gmsh,
    reason="This test requires Gmsh.",
)
def test_remeshing(copy_test_asset, config_dir, work_dir):
    remesh_dir = copy_test_asset("data/meshes/remesh")
    mesh_file = remesh_dir / "mesh.xdmf"
    result_dir = work_dir / "temp"

    def mesh_parametrization(mesh_file):
        config = cashocs.load_config(str(config_dir / "config_remesh.ini"))
        config["Mesh"]["gmsh_file"] = str(remesh_dir / "mesh.msh")
        config["Mesh"]["geo_file"] = str(remesh_dir / "mesh.geo")
        config["Output"]["result_dir"] = str(result_dir)
        config["Debug"]["remeshing"] = True

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(mesh_file)

        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        p = Function(V)

        x = SpatialCoordinate(mesh)
        f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

        e = inner(grad(u), grad(p)) * dx - f * p * dx
        bcs = DirichletBC(V, Constant(0), boundaries, 1)

        J = cashocs.IntegralFunctional(u * dx)

        args = (e, bcs, J, u, p, boundaries, config)
        kwargs = {}

        return args, kwargs

    sop = cashocs.ShapeOptimizationProblem(mesh_parametrization, str(mesh_file))
    sop.solve()

    MPI.barrier(MPI.comm_world)
    assert any(
        folder.name.startswith("cashocs_remesh_") for folder in remesh_dir.iterdir()
    )

    assert result_dir.is_dir()
    assert (result_dir / "checkpoints").is_dir()
    assert (result_dir / "history.txt").is_file()
    assert (result_dir / "history.json").is_file()
    assert (result_dir / "optimized_mesh.msh").is_file()
    assert (result_dir / "checkpoints/adjoint_0.xdmf").is_file()
    assert (result_dir / "checkpoints/adjoint_0.h5").is_file()
    assert (result_dir / "checkpoints/state_0.xdmf").is_file()
    assert (result_dir / "checkpoints/state_0.h5").is_file()
    assert (result_dir / "checkpoints/shape_gradient.xdmf").is_file()
    assert (result_dir / "checkpoints/shape_gradient.h5").is_file()

    MPI.barrier(MPI.comm_world)


@pytest.mark.slow
@pytest.mark.skipif(
    not has_gmsh,
    reason="This test requires Gmsh.",
)
def test_remeshing_with_quantile_quality(copy_test_asset, config_dir, work_dir):
    remesh_dir = copy_test_asset("data/meshes/remesh")
    mesh_file = remesh_dir / "mesh.xdmf"
    result_dir = work_dir / "temp"

    def mesh_parametrization(mesh_file):
        config = cashocs.load_config(str(config_dir / "config_remesh.ini"))
        config["Mesh"]["gmsh_file"] = str(remesh_dir / "mesh.msh")
        config["Mesh"]["geo_file"] = str(remesh_dir / "mesh.geo")
        config["Output"]["result_dir"] = str(result_dir)
        config["MeshQuality"]["type"] = "quantile"
        config["MeshQuality"]["quantile"] = 0.05
        config["MeshQuality"]["tol_lower"] = 0.4
        config["MeshQuality"]["tol_upper"] = 0.45
        config["Debug"]["remeshing"] = True

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(mesh_file)

        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        p = Function(V)

        x = SpatialCoordinate(mesh)
        f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

        e = inner(grad(u), grad(p)) * dx - f * p * dx
        bcs = DirichletBC(V, Constant(0), boundaries, 1)

        J = cashocs.IntegralFunctional(u * dx)

        args = (e, bcs, J, u, p, boundaries, config)
        kwargs = {}

        return args, kwargs

    sop = cashocs.ShapeOptimizationProblem(mesh_parametrization, str(mesh_file))
    sop.solve()

    MPI.barrier(MPI.comm_world)
    assert any(
        folder.name.startswith("cashocs_remesh_") for folder in remesh_dir.iterdir()
    )

    assert result_dir.is_dir()
    assert (result_dir / "checkpoints").is_dir()
    assert (result_dir / "history.txt").is_file()
    assert (result_dir / "history.json").is_file()
    assert (result_dir / "optimized_mesh.msh").is_file()
    assert (result_dir / "checkpoints/adjoint_0.xdmf").is_file()
    assert (result_dir / "checkpoints/adjoint_0.h5").is_file()
    assert (result_dir / "checkpoints/state_0.xdmf").is_file()
    assert (result_dir / "checkpoints/state_0.h5").is_file()
    assert (result_dir / "checkpoints/shape_gradient.xdmf").is_file()
    assert (result_dir / "checkpoints/shape_gradient.h5").is_file()

    MPI.barrier(MPI.comm_world)


def test_remesh_scaling(copy_test_asset, config_dir, work_dir):
    remesh_dir = copy_test_asset("data/meshes/remesh")
    result_dir = work_dir / "temp"
    rng = np.random.RandomState(300696)
    w_des = rng.rand(1)[0]
    mesh_file = remesh_dir / "mesh.xdmf"

    def mesh_parametrization(mesh_file):
        config = cashocs.load_config(str(config_dir / "config_remesh.ini"))

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(mesh_file))
        config["Mesh"]["gmsh_file"] = str(remesh_dir / "mesh.msh")
        config["Mesh"]["geo_file"] = str(remesh_dir / "mesh.geo")
        config["Output"]["result_dir"] = str(result_dir)

        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        p = Function(V)

        x = SpatialCoordinate(mesh)
        f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

        e = inner(grad(u), grad(p)) * dx - f * p * dx
        bcs = DirichletBC(V, Constant(0), boundaries, 1)

        J = cashocs.IntegralFunctional(u * dx)

        args = (e, bcs, J, u, p, boundaries, config)
        kwargs = {"desired_weights": [w_des]}

        return args, kwargs

    sop = cashocs.ShapeOptimizationProblem(mesh_parametrization, str(mesh_file))
    val = sop.reduced_cost_functional.evaluate()
    assert np.abs(np.abs(val) - w_des) < 1e-14

    MPI.barrier(MPI.comm_world)
    MPI.barrier(MPI.comm_world)


def test_remeshing_disabled(copy_test_asset, config_dir):
    remesh_dir = copy_test_asset("data/meshes/remesh")
    mesh_file = remesh_dir / "mesh.xdmf"

    config = cashocs.load_config(str(config_dir / "config_remesh.ini"))
    config["Mesh"]["gmsh_file"] = str(remesh_dir / "mesh.msh")
    config["Mesh"]["geo_file"] = str(remesh_dir / "mesh.geo")
    config["Mesh"]["remesh"] = False
    config["Output"]["save_state"] = False
    config["Output"]["save_adjoint"] = False
    config["Output"]["save_gradient"] = False
    config["Output"]["save_mesh"] = False
    config["Output"]["save_results"] = False
    config["Output"]["save_txt"] = False
    config["OptimizationRoutine"]["soft_exit"] = False

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(str(mesh_file))

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    p = Function(V)

    x = SpatialCoordinate(mesh)
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    e = inner(grad(u), grad(p)) * dx - f * p * dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

    J = cashocs.IntegralFunctional(u * dx)

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve(max_iter=7)
