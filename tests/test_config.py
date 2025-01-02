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

from collections import namedtuple

from fenics import *
import pytest

import cashocs
from cashocs._exceptions import ConfigError


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


@pytest.fixture
def ksp_options():
    return {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 0.0,
        "ksp_atol": 0.0,
        "ksp_max_it": 1,
        "ksp_monitor_true_residual": None,
    }


@pytest.fixture
def ocp(F, bcs, J, y, u, p, config):
    return cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)


@pytest.fixture
def ocp_ksp(F, bcs, J, y, u, p, config, ksp_options):
    return cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p, config=config, ksp_options=ksp_options
    )


def test_correct_config(dir_path):
    cashocs.load_config(f"{dir_path}/config_ocp.ini")
    cashocs.load_config(f"{dir_path}/config_sop.ini")
    cashocs.load_config(f"{dir_path}/config_picard.ini")
    cashocs.load_config(f"{dir_path}/config_remesh.ini")

    assert 1 == 1


def test_config_error(dir_path, F, bcs, J, y, u, p):
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
        config.set("Mesh", "remesh", "1.0")

        cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)

    assert "You have some error(s) in your config file" in str(e_info.value)
    assert (
        "Key remesh in section Mesh has the wrong type. Required type is bool."
        in str(e_info.value)
    )


def test_incorrect_configs(dir_path, F, bcs, J, y, u, p):
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/test_config.ini")
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)

    assert (
        "The following section is not valid: <Section: A>\nThe following section is not valid: <Section: B>\nKey algorithm is not valid for section StateSystem."
        in str(e_info.value)
    )


def test_incompatible_config(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("MeshQuality", "tol_lower", "0.5")
    config_sop.set("MeshQuality", "tol_upper", "0.1")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "The value of key tol_upper in section MeshQuality is smaller than the value of key tol_lower in section MeshQuality, but it should be larger."
        in str(e_info.value)
    )


def test_larger_than_dependency_config(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("ShapeGradient", "dist_max", "0.5")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "The value of key dist_max in section ShapeGradient is smaller than the value of key dist_min in section ShapeGradient, but it should be larger."
        in str(e_info.value)
    )


def test_larger_equal_than_dependency_config(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("Regularization", "x_end", "-1.0")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "The value of key x_end in section Regularization is smaller than the value of key x_start in section Regularization, but it should be larger."
        in str(e_info.value)
    )


def test_file_extension(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("Mesh", "geo_file", "test.msh")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key geo_file in section Mesh has the wrong file extension, it should end in .geo."
        in str(e_info.value)
    )


def test_non_negative_attribute(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("MeshQuality", "tol_lower", "-1e-1")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key tol_lower in section MeshQuality is negative, but it must not be."
        in str(e_info.value)
    )


def test_positive_attribute(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("MeshQuality", "tol_upper", "0.0")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key tol_upper in section MeshQuality is non-positive, but it most be positive."
        in str(e_info.value)
    )


def test_less_than_one_attribute(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("MeshQuality", "tol_upper", "2.0")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key tol_upper in section MeshQuality is larger than one, but it must be smaller."
        in str(e_info.value)
    )


def test_larger_than_one_attribute(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("MeshQuality", "volume_change", "0.5")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key volume_change in section MeshQuality is smaller than one, but it must be larger."
        in str(e_info.value)
    )


def test_possible_options(config_sop, F, bcs, J, y, p, geometry):
    config_sop.set("OptimizationRoutine", "gradient_method", "mymethod")
    with pytest.raises(ConfigError) as e_info:
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key gradient_method in section OptimizationRoutine has a wrong value. Possible options are ['direct', 'iterative']."
        in str(e_info.value)
    )


def test_incomplete_requirements_config(config_sop, F, bcs, J, y, p, geometry):
    with pytest.raises(ConfigError) as e_info:
        config_sop.set("Output", "save_mesh", "True")
        cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, geometry.boundaries, config=config_sop
        )

    assert (
        "Key save_mesh in section Output requires key gmsh_file in section Mesh to be present."
        in str(e_info.value)
    )
