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

"""Tests for the geometry module."""

import pathlib
import subprocess
import sys

import fenics
from mpi4py import MPI
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs._exceptions import InputError
from cashocs.geometry import MeshQuality
from cashocs.io.mesh import gather_coordinates


@pytest.fixture
def regular_mesh():
    return cashocs.regular_mesh(5)[0]


@pytest.fixture
def unit_square_mesh():
    return fenics.UnitSquareMesh(5, 5)


@pytest.fixture
def interval_mesh():
    return fenics.UnitIntervalMesh(10)


def test_mesh_import(dir_path):
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

    assert fenics.assemble(1 * dx) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds) == pytest.approx(4.0, rel=1e-14)

    assert fenics.assemble(1 * ds(1)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1.0, rel=1e-14)
    assert fenics.assemble(1 * ds(4)) == pytest.approx(1.0, rel=1e-14)

    fe_coords = gather_coordinates(mesh)
    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(fe_coords, gmsh_coords)
    MPI.COMM_WORLD.barrier()

    assert pathlib.Path(f"{dir_path}/mesh/mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/mesh_boundaries.h5").is_file()
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
    MPI.COMM_WORLD.barrier()


def test_regular_mesh(rng, unit_square_mesh, regular_mesh):
    lens = rng.uniform(0.5, 2, 2)
    r_mesh, _, _, _, _, _ = cashocs.regular_mesh(2, lens[0], lens[1])

    max_vals = rng.uniform(0.5, 1, 3)
    min_vals = rng.uniform(-1, -0.5, 3)

    s_mesh, _, _, _, _, _ = cashocs.regular_box_mesh(
        2, min_vals[0], min_vals[1], min_vals[2], max_vals[0], max_vals[1], max_vals[2]
    )

    u_coords = gather_coordinates(unit_square_mesh)
    c_coords = gather_coordinates(regular_mesh)
    r_coords = gather_coordinates(r_mesh)
    s_coords = gather_coordinates(s_mesh)

    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(c_coords, u_coords)

        assert np.all((np.max(r_coords, axis=0) - lens) < 1e-14)
        assert np.all((np.min(r_coords, axis=0) - np.array([0, 0])) < 1e-14)

        assert np.all(abs(np.max(s_coords, axis=0) - max_vals) < 1e-14)
        assert np.all(abs(np.min(s_coords, axis=0) - min_vals) < 1e-14)
    MPI.COMM_WORLD.barrier()

    t_mesh, _, _, _, _, _ = cashocs.regular_box_mesh(
        2, start_x=0.0, end_x=lens[0], start_y=0.0, end_y=lens[1]
    )
    t_coords = gather_coordinates(t_mesh)

    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(t_coords, r_coords)
    MPI.COMM_WORLD.barrier()


def test_mesh_quality_2D():
    mesh, _, _, _, _, _ = cashocs.regular_mesh(4)

    opt_angle = 60 / 360 * 2 * np.pi
    alpha_1 = 90 / 360 * 2 * np.pi
    alpha_2 = 45 / 360 * 2 * np.pi

    q_1 = 1 - np.maximum(
        (alpha_1 - opt_angle) / (np.pi - opt_angle), (opt_angle - alpha_1) / (opt_angle)
    )
    q_2 = 1 - np.maximum(
        (alpha_2 - opt_angle) / (np.pi - opt_angle), (opt_angle - alpha_2) / (opt_angle)
    )
    q = np.minimum(q_1, q_2)

    min_max_angle = cashocs.compute_mesh_quality(mesh, "min", "maximum_angle")
    min_radius_ratios = cashocs.compute_mesh_quality(mesh, "min", "radius_ratios")
    avg_radius_ratios = cashocs.compute_mesh_quality(mesh, "avg", "radius_ratios")
    q_radius_ratios = cashocs.compute_mesh_quality(
        mesh, "quantile", "radius_ratios", quantile=0.351
    )
    min_condition = cashocs.compute_mesh_quality(mesh, "min", "condition_number")
    avg_condition = cashocs.compute_mesh_quality(mesh, "avg", "condition_number")
    q_condition = cashocs.compute_mesh_quality(
        mesh, "quantile", "condition_number", quantile=0.75189
    )

    assert (
        abs(min_max_angle - cashocs.compute_mesh_quality(mesh, "avg", "maximum_angle"))
        < 1e-14
    )
    assert abs(min_max_angle - q) < 1e-14
    assert (
        abs(min_max_angle - cashocs.compute_mesh_quality(mesh, "avg", "skewness"))
        < 1e-14
    )
    assert (
        abs(min_max_angle - cashocs.compute_mesh_quality(mesh, "min", "skewness"))
        < 1e-14
    )
    assert (
        abs(
            min_max_angle
            - cashocs.compute_mesh_quality(mesh, "quantile", "skewness", quantile=0.167)
        )
        < 1e-14
    )

    assert abs(min_radius_ratios - avg_radius_ratios) < 1e-14
    assert abs(min_radius_ratios - q_radius_ratios) < 1e-14
    assert (
        abs(min_radius_ratios - np.min(fenics.MeshQuality.radius_ratio_min_max(mesh)))
        < 1e-14
    )

    assert abs(min_condition - avg_condition) < 1e-14
    assert abs(min_condition - q_condition) < 1e-14
    assert abs(min_condition - 0.4714045207910318) < 1e-14


def test_mesh_quality_3D():
    mesh, _, _, _, _, _ = cashocs.regular_mesh(4, 1.0, 1.0, 1.0)
    opt_angle = np.arccos(1 / 3)
    dh_min_max = fenics.MeshQuality.dihedral_angles_min_max(mesh)
    alpha_min = dh_min_max[0]
    alpha_max = dh_min_max[1]

    q_1 = 1 - np.maximum(
        (alpha_max - opt_angle) / (np.pi - opt_angle),
        (opt_angle - alpha_max) / (opt_angle),
    )
    q_2 = 1 - np.maximum(
        (alpha_min - opt_angle) / (np.pi - opt_angle),
        (opt_angle - alpha_min) / (opt_angle),
    )
    q = np.minimum(q_1, q_2)

    r_1 = 1 - np.maximum((alpha_max - opt_angle) / (np.pi - opt_angle), 0.0)
    r_2 = 1 - np.maximum((alpha_min - opt_angle) / (np.pi - opt_angle), 0.0)
    r = np.minimum(r_1, r_2)

    min_max_angle = cashocs.compute_mesh_quality(mesh, "min", "maximum_angle")
    min_radius_ratios = cashocs.compute_mesh_quality(mesh, "min", "radius_ratios")
    avg_radius_ratios = cashocs.compute_mesh_quality(mesh, "avg", "radius_ratios")
    q_radius_ratios = cashocs.compute_mesh_quality(
        mesh, "quantile", "radius_ratios", quantile=0.615
    )
    min_condition = cashocs.compute_mesh_quality(mesh, "min", "condition_number")
    avg_condition = cashocs.compute_mesh_quality(mesh, "avg", "condition_number")
    q_condition = cashocs.compute_mesh_quality(
        mesh, "quantile", "condition_number", quantile=0.91576
    )
    min_skewness = cashocs.compute_mesh_quality(mesh, "min", "skewness")

    assert (
        abs(min_max_angle - cashocs.compute_mesh_quality(mesh, "avg", "maximum_angle"))
        < 1e-14
    )
    assert (
        abs(
            min_max_angle
            - cashocs.compute_mesh_quality(
                mesh, "quantile", "maximum_angle", quantile=0.715
            )
        )
        < 1e-14
    )
    assert abs(min_max_angle - r) < 1e-14

    assert (
        abs(min_skewness - cashocs.compute_mesh_quality(mesh, "avg", "skewness"))
        < 1e-14
    )
    assert (
        abs(
            min_skewness
            - cashocs.compute_mesh_quality(mesh, "quantile", "skewness", quantile=0.14)
        )
        < 1e-14
    )
    assert abs(min_skewness - q) < 1e-14

    assert abs(min_radius_ratios - avg_radius_ratios) < 1e-14
    assert abs(min_radius_ratios - q_radius_ratios) < 1e-14
    assert (
        abs(min_radius_ratios - np.min(fenics.MeshQuality.radius_ratio_min_max(mesh)))
        < 1e-14
    )

    assert abs(min_condition - avg_condition) < 1e-14
    assert abs(min_condition - q_condition) < 1e-14
    assert abs(min_condition - 0.3162277660168379) < 1e-14


def test_write_mesh():
    dir_path = str(pathlib.Path(__file__).parent)
    cashocs.convert(f"{dir_path}/mesh/mesh.msh", f"{dir_path}/mesh/mesh.xdmf")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/mesh.xdmf"
    )

    cashocs.io.write_out_mesh(
        mesh, dir_path + "/mesh/mesh.msh", dir_path + "/mesh/test.msh"
    )

    cashocs.convert(f"{dir_path}/mesh/test.msh", f"{dir_path}/mesh/test.xdmf")
    test, _, _, _, _, _ = cashocs.import_mesh(dir_path + "/mesh/test.xdmf")

    test_coords = gather_coordinates(test)
    mesh_coords = gather_coordinates(mesh)
    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(test_coords, mesh_coords)

        subprocess.run(["rm", f"{dir_path}/mesh/test.msh"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/test_boundaries.h5"], check=True)

        subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
    MPI.COMM_WORLD.barrier()


def test_empty_measure(rng):
    mesh, _, _, dx, ds, dS = cashocs.regular_mesh(5)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    dm = cashocs.geometry._EmptyMeasure(dx)

    trial = fenics.TrialFunction(V)
    test = fenics.TestFunction(V)

    assert fenics.assemble(1 * dm) == 0.0
    assert (fenics.assemble(test * dm).norm("linf")) == 0.0
    assert (fenics.assemble(trial * test * dm).norm("linf")) == 0.0

    fun = fenics.Function(V)
    fun.vector().set_local(rng.rand(fun.vector().local_size()))
    fun.vector().apply("")

    d1 = cashocs.geometry._EmptyMeasure(dx)
    d2 = cashocs.geometry._EmptyMeasure(ds)

    a = trial * test * d1 + trial * test * d2
    L = fun * test * d1 + fun * test * d2
    F = fun * d1 + fun * d2

    A = fenics.assemble(a)
    b = fenics.assemble(L)
    c = fenics.assemble(F)

    assert np.max(np.abs(A.array())) == 0.0
    assert np.max(np.abs(b[:])) == 0.0
    assert c == 0.0

    dE = cashocs.geometry.generate_measure([], dx)
    assert fenics.assemble(fenics.Constant(1) * dE) == 0.0


def test_convert_coordinate_defo_to_dof_defo():
    mesh, _, _, _, _, _ = cashocs.regular_mesh(20)
    coordinates_initial = mesh.coordinates().copy()
    a_priori_tester = cashocs.geometry.mesh_testing.APrioriMeshTester(mesh)
    intersection_tester = cashocs.geometry.mesh_testing.IntersectionTester(mesh)
    deformation_handler = cashocs.geometry.DeformationHandler(
        mesh, a_priori_tester, intersection_tester
    )
    VCG = fenics.VectorFunctionSpace(mesh, "CG", 1)
    coordinate_deformation = (
        fenics.interpolate(fenics.Expression(("x[0]", "x[1]"), degree=1), VCG)
        .compute_vertex_values()
        .reshape(2, -1)
        .T
    )

    coordinates_transformed = coordinates_initial + coordinate_deformation

    vector_field = deformation_handler.coordinate_to_dof(coordinate_deformation)
    assert deformation_handler.move_mesh(vector_field)
    assert np.max(np.abs(mesh.coordinates()[:, :] - coordinates_transformed)) <= 1e-15


def test_convert_dof_defo_to_coordinate_defo(rng):
    mesh, _, _, _, _, _ = cashocs.regular_mesh(20)
    coordinates_initial = mesh.coordinates().copy()
    a_priori_tester = cashocs.geometry.mesh_testing.APrioriMeshTester(mesh)
    intersection_tester = cashocs.geometry.mesh_testing.IntersectionTester(mesh)
    deformation_handler = cashocs.geometry.DeformationHandler(
        mesh, a_priori_tester, intersection_tester
    )
    VCG = fenics.VectorFunctionSpace(mesh, "CG", 1)
    defo = fenics.Function(VCG)
    dof_vector = rng.randn(defo.vector().local_size())
    h = mesh.hmin()
    dof_vector *= h / (4.0 * np.max(np.abs(dof_vector)))
    defo = fenics.Function(VCG)
    defo.vector().set_local(dof_vector)
    defo.vector().apply("")

    coordinate_deformation = deformation_handler.dof_to_coordinate(defo)
    coordinates_transformed = coordinates_initial + coordinate_deformation
    assert deformation_handler.move_mesh(defo)
    assert np.max(np.abs(mesh.coordinates()[:, :] - coordinates_transformed)) <= 1e-15


def test_move_mesh():
    mesh, _, _, _, _, _ = cashocs.regular_mesh(20)
    coordinates_initial = mesh.coordinates().copy()
    a_priori_tester = cashocs.geometry.mesh_testing.APrioriMeshTester(mesh)
    intersection_tester = cashocs.geometry.mesh_testing.IntersectionTester(mesh)
    deformation_handler = cashocs.geometry.DeformationHandler(
        mesh, a_priori_tester, intersection_tester
    )

    coordinate_deformation = coordinates_initial.copy()
    h = mesh.hmin()
    coordinate_deformation *= h / (4.0 * 0.09)

    coords_added = coordinates_initial + coordinate_deformation
    mesh.coordinates()[:, :] = coords_added
    coordinates_added = gather_coordinates(mesh)
    mesh.coordinates()[:, :] = coordinates_initial
    mesh.bounding_box_tree().build(mesh)

    assert deformation_handler.move_mesh(coordinate_deformation)
    coordinates_moved = gather_coordinates(mesh)

    deformation_handler.revert_transformation()

    MPI.COMM_WORLD.barrier()

    vector_field = deformation_handler.coordinate_to_dof(coordinate_deformation)
    assert deformation_handler.move_mesh(vector_field)
    coordinates_dof_moved = gather_coordinates(mesh)

    if MPI.COMM_WORLD.rank == 0:
        assert np.max(np.abs(coordinates_added - coordinates_moved)) <= 1e-15
        assert np.max(np.abs(coordinates_dof_moved - coordinates_added)) <= 1e-15
        assert np.max(np.abs(coordinates_dof_moved - coordinates_moved)) <= 1e-15
    MPI.COMM_WORLD.barrier()


@pytest.mark.parametrize(
    "method, expected_tolerance", [("poisson", 0.28), ("eikonal", 5e-2)]
)
def test_boundary_distance(method, expected_tolerance):
    mesh, _, boundaries, _, _, _ = cashocs.regular_mesh(16)
    dist = cashocs.geometry.compute_boundary_distance(
        mesh, boundaries=boundaries, boundary_idcs=[1, 2, 3, 4], method=method
    )
    assert dist.vector().min() >= 0.0
    assert np.abs(dist.vector().max() - 0.5) / 0.5 <= expected_tolerance

    dist = cashocs.geometry.compute_boundary_distance(
        mesh, boundaries=boundaries, boundary_idcs=[1], method=method
    )
    assert dist.vector().min() >= 0.0
    assert np.abs(dist.vector().max() - 1.0) / 1.0 <= expected_tolerance

    dist = cashocs.geometry.compute_boundary_distance(mesh, method=method)
    assert dist.vector().min() >= 0.0
    assert np.abs(dist.vector().max() - 0.5) / 0.5 <= expected_tolerance


def test_named_mesh_import():
    dir_path = str(pathlib.Path(__file__).parent)
    cashocs.convert(
        f"{dir_path}/mesh/named_mesh.msh", f"{dir_path}/mesh/named_mesh.xdmf"
    )

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        f"{dir_path}/mesh/named_mesh.xdmf"
    )

    assert fenics.assemble(1 * dx("volume")) == fenics.assemble(1 * dx(1))
    assert fenics.assemble(1 * ds("inlet")) == fenics.assemble(1 * ds(1))
    assert fenics.assemble(1 * ds("wall")) == fenics.assemble(1 * ds(2))
    assert fenics.assemble(1 * ds("outlet")) == fenics.assemble(1 * ds(3))

    assert dx("volume") == dx(1)
    assert ds("inlet") == ds(1)
    assert ds("wall") == ds(2)
    assert ds("outlet") == ds(3)

    with pytest.raises(InputError) as e_info:
        dx("inlet")
        assert "subdomain_id" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ds("volume")
        assert "subdomain_id" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        dx("fantasy")
        assert "subdomain_id" in str(e_info.value)

    assert pathlib.Path(f"{dir_path}/mesh/named_mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_boundaries.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_physical_groups.json").is_file()

    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh.h5"], check=True)
        subprocess.run(
            ["rm", f"{dir_path}/mesh/named_mesh_boundaries.xdmf"], check=True
        )
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_boundaries.h5"], check=True)
        subprocess.run(
            ["rm", f"{dir_path}/mesh/named_mesh_physical_groups.json"], check=True
        )
    MPI.COMM_WORLD.barrier()


def test_legacy_mesh_import():
    dir_path = str(pathlib.Path(__file__).parent)

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        f"{dir_path}/mesh/physical_names_legacy/named_mesh.xdmf"
    )

    assert fenics.assemble(1 * dx) == pytest.approx(1.0)
    assert fenics.assemble(1 * ds) == pytest.approx(4.0)

    assert fenics.assemble(1 * dx(1)) == pytest.approx(1.0)
    assert fenics.assemble(1 * ds(1)) == pytest.approx(1.0)
    assert fenics.assemble(1 * ds(2)) == pytest.approx(2.0)
    assert fenics.assemble(1 * ds(3)) == pytest.approx(1.0)

    assert fenics.assemble(1 * dx("volume")) == pytest.approx(1.0)
    assert fenics.assemble(1 * ds("inlet")) == pytest.approx(1.0)
    assert fenics.assemble(1 * ds("wall")) == pytest.approx(2.0)
    assert fenics.assemble(1 * ds("outlet")) == pytest.approx(1.0)

    assert dx("volume") == dx(1)
    assert ds("inlet") == ds(1)
    assert ds("wall") == ds(2)
    assert ds("outlet") == ds(3)

    with pytest.raises(InputError) as e_info:
        dx("inlet")
        assert "subdomain_id" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ds("volume")
        assert "subdomain_id" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        dx("fantasy")
        assert "subdomain_id" in str(e_info.value)


def test_create_measure():
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
    V = fenics.FunctionSpace(mesh, "CG", 1)

    meas = cashocs.geometry.generate_measure([1, 2, 3], ds)
    test = ds(1) + ds(2) + ds(3)

    assert abs(fenics.assemble(1 * meas) - 3) < 1e-14
    for i in range(3):
        assert meas._measures[i] == test._measures[i]


def test_list_measure():
    mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)

    m_sum = ds(1) + ds(2) + ds(3)
    ref = ds([1, 2, 3])

    for i in range(3):
        assert m_sum._measures[i] == ref._measures[i]


def test_interval_mesh(interval_mesh, rng):
    mesh, _, _, _, _, _ = cashocs.interval_mesh(10)
    coords = gather_coordinates(mesh)
    i_coords = gather_coordinates(interval_mesh)
    if MPI.COMM_WORLD.rank == 0:
        assert np.allclose(coords, i_coords)
    MPI.COMM_WORLD.barrier()

    lens = rng.uniform(0.5, 2, 2)
    lens.sort()
    mesh, _, _, _, _, _ = cashocs.interval_mesh(10, lens[0], lens[1])
    expr = fenics.Expression("x[0]", degree=1)
    fun = fenics.interpolate(expr, fenics.FunctionSpace(mesh, "CG", 1))

    assert abs(fun.vector().max() - lens[1]) <= 1e-15
    assert abs(fun.vector().min() - lens[0]) <= 1e-15

    partitions = rng.uniform(0.1, 0.9, 5)
    partitions = partitions.round(2)
    partitions.sort()
    mesh, _, _, dx, _, _ = cashocs.interval_mesh(100, 0, 1, partitions)

    assert abs(fenics.assemble(1 * dx(1)) - partitions[0]) <= 1e-15
    assert abs(fenics.assemble(1 * dx(2)) - partitions[1] + partitions[0]) <= 1e-15
    assert abs(fenics.assemble(1 * dx(3)) - partitions[2] + partitions[1]) <= 1e-15
    assert abs(fenics.assemble(1 * dx(4)) - partitions[3] + partitions[2]) <= 1e-15
    assert abs(fenics.assemble(1 * dx(5)) - partitions[4] + partitions[3]) <= 1e-15
    assert abs(fenics.assemble(1 * dx(6)) - 1.0 + partitions[4]) <= 1e-15
