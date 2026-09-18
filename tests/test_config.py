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


from fenics import *
import pytest

import cashocs
from cashocs._exceptions import ConfigError


def test_correct_config(dir_path):
    config = cashocs.load_config(f"{dir_path}/config_ocp.ini")
    config.validate_config()

    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.validate_config()

    config = cashocs.load_config(f"{dir_path}/config_picard.ini")
    config.validate_config()

    assert 1 == 1


def test_config_error(dir_path):
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/config_sop.ini")
        config["Mesh"]["remesh"] = "1.0"
        config.validate_config()

    assert "You have some error(s) in your config file" in str(e_info.value)
    assert (
        "Option remesh in section Mesh has the wrong type. Required type is bool."
        in str(e_info.value)
    )


def test_incorrect_configs(dir_path):
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/test_config.ini")
        config.validate_config()

    assert (
        "The following section is not valid: A\nThe following section is not valid: B\nOption algorithm not valid for section StateSystem."
        in str(e_info.value)
    )


def test_incompatible_config(config_sop):
    config_sop.set("MeshQuality", "tol_lower", "0.5")
    config_sop.set("MeshQuality", "tol_upper", "0.1")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "The value of option tol_upper in section MeshQuality is smaller than the value of option tol_lower in section MeshQuality, but it should be larger."
        in str(e_info.value)
    )


def test_larger_than_dependency_config(config_sop):
    config_sop.set("ShapeGradient", "dist_max", "0.5")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "The value of option dist_max in section ShapeGradient is smaller than the value of option dist_min in section ShapeGradient, but it should be larger."
        in str(e_info.value)
    )


def test_larger_equal_than_dependency_config(config_sop):
    config_sop.set("Regularization", "x_end", "-1.0")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "The value of option x_end in section Regularization is smaller than the value of option x_start in section Regularization, but it should be larger."
        in str(e_info.value)
    )


def test_file_extension(config_sop):
    config_sop.set("Mesh", "geo_file", "test.msh")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "Option geo_file in section Mesh has the wrong file extension, it should end in .geo."
        in str(e_info.value)
    )


def test_non_negative_attribute(config_sop):
    config_sop.set("MeshQuality", "tol_lower", "-1e-1")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "Option tol_lower in section MeshQuality is negative, but it must not be."
        in str(e_info.value)
    )


def test_positive_attribute(config_sop):
    config_sop.set("MeshQuality", "tol_upper", "0.0")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "Option tol_upper in section MeshQuality is non-positive, but it most be positive."
        in str(e_info.value)
    )


def test_less_than_one_attribute(config_sop):
    config_sop.set("MeshQuality", "tol_upper", "2.0")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "Option tol_upper in section MeshQuality is larger than one, but it must be smaller."
        in str(e_info.value)
    )


def test_larger_than_one_attribute(config_sop):
    config_sop.set("MeshQuality", "volume_change", "0.5")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "Option volume_change in section MeshQuality is smaller than one, but it must be larger."
        in str(e_info.value)
    )


def test_possible_options(config_sop):
    config_sop.set("OptimizationRoutine", "gradient_method", "mymethod")
    with pytest.raises(ConfigError) as e_info:
        config_sop.validate_config()

    assert (
        "Option gradient_method in section OptimizationRoutine has a wrong value. Possible options are ['direct', 'iterative']."
        in str(e_info.value)
    )


def test_incomplete_requirements_config(config_sop):
    with pytest.raises(ConfigError) as e_info:
        config_sop.set("Output", "save_mesh", "True")
        config_sop.validate_config()

    assert (
        "Option save_mesh in section Output requires option gmsh_file in section Mesh to be present."
        in str(e_info.value)
    )
