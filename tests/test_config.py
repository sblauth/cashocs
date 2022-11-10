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


import os
import pathlib
import subprocess

from fenics import *
import numpy as np
import pytest

import cashocs
from cashocs._exceptions import ConfigError
from cashocs._exceptions import InputError

rng = np.random.RandomState(300696)
dir_path = str(pathlib.Path(__file__).parent)
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

ksp_options = [
    ["ksp_type", "cg"],
    ["pc_type", "hypre"],
    ["pc_hypre_type", "boomeramg"],
    ["ksp_rtol", 0.0],
    ["ksp_atol", 0.0],
    ["ksp_max_it", 1],
    ["ksp_monitor_true_residual"],
]

ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
ocp_ksp = cashocs.OptimalControlProblem(
    F, bcs, J, y, u, p, config, ksp_options=ksp_options
)


def test_correct_config():
    config = cashocs.load_config(f"{dir_path}/config_ocp.ini")
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config = cashocs.load_config(f"{dir_path}/config_picard.ini")
    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")

    assert 1 == 1


def test_config_error():
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
        config.set("Mesh", "remesh", "1.0")

        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)

    assert "You have some error(s) in your config file" in str(e_info.value)
    assert (
        "Key remesh in section Mesh has the wrong type. Required type is bool."
        in str(e_info.value)
    )


def test_incorrect_configs():
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/test_config.ini")
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)

    assert (
        "The following section is not valid: <Section: A>\nThe following section is not valid: <Section: B>\nKey algorithm is not valid for section StateSystem."
        in str(e_info.value)
    )


def test_incompatible_config():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("MeshQuality", "tol_lower", "0.5")
    config.set("MeshQuality", "tol_upper", "0.1")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "The value of key tol_upper in section MeshQuality is smaller than the value of key tol_lower in section MeshQuality, but it should be larger."
        in str(e_info.value)
    )


def test_larger_than_dependency_config():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("ShapeGradient", "dist_max", "0.5")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "The value of key dist_max in section ShapeGradient is smaller than the value of key dist_min in section ShapeGradient, but it should be larger."
        in str(e_info.value)
    )


def test_larger_equal_than_dependency_config():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("Regularization", "x_end", "-1.0")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "The value of key x_end in section Regularization is smaller than the value of key x_start in section Regularization, but it should be larger."
        in str(e_info.value)
    )


def test_file_extension():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("Mesh", "geo_file", "test.msh")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key geo_file in section Mesh has the wrong file extension, it should end in .geo."
        in str(e_info.value)
    )


def test_non_negative_attribute():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("MeshQuality", "tol_lower", "-1e-1")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key tol_lower in section MeshQuality is negative, but it must not be."
        in str(e_info.value)
    )


def test_positive_attribute():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("MeshQuality", "tol_upper", "0.0")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key tol_upper in section MeshQuality is non-positive, but it most be positive."
        in str(e_info.value)
    )


def test_less_than_one_attribute():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("MeshQuality", "tol_upper", "2.0")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key tol_upper in section MeshQuality is larger than one, but it must be smaller."
        in str(e_info.value)
    )


def test_larger_than_one_attribute():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("MeshQuality", "volume_change", "0.5")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key volume_change in section MeshQuality is smaller than one, but it must be larger."
        in str(e_info.value)
    )


def test_possible_options():
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config.set("OptimizationRoutine", "gradient_method", "mymethod")
    with pytest.raises(ConfigError) as e_info:
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key gradient_method in section OptimizationRoutine has a wrong value. Possible options are ['direct', 'iterative']."
        in str(e_info.value)
    )


def test_incomplete_requirements_config():
    with pytest.raises(ConfigError) as e_info:
        config = cashocs.load_config(f"{dir_path}/config_sop.ini")
        config.set("Output", "save_mesh", "True")
        sop = cashocs.ShapeOptimizationProblem(
            F, bcs, J, y, p, boundaries, config=config
        )

    assert (
        "Key save_mesh in section Output requires key gmsh_file in section Mesh to be present."
        in str(e_info.value)
    )


def test_no_config():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    cwd = pathlib.Path.cwd()
    try:
        if MPI.rank(MPI.comm_world) == 0:
            os.chdir("./tests")
        MPI.barrier(MPI.comm_world)
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p)
        with pytest.raises(InputError) as e_info:
            ocp.solve(rtol=1e-2, atol=0.0, max_iter=7)
        assert "You did not specify a solution algorithm in your config file." in str(
            e_info.value
        )
        ocp.solve(algorithm="bfgs", rtol=1e-2, atol=0.0, max_iter=7)
        assert ocp.solver.relative_norm <= ocp.solver.rtol

        MPI.barrier(MPI.comm_world)
        assert pathlib.Path(dir_path + "/results").is_dir()
        assert pathlib.Path(dir_path + "/results/history.txt").is_file()
        assert pathlib.Path(dir_path + "/results/history.json").is_file()

        MPI.barrier(MPI.comm_world)
        if MPI.rank(MPI.comm_world) == 0:
            subprocess.run(["rm", "-r", f"{dir_path}/results"], check=True)
        MPI.barrier(MPI.comm_world)
    except:
        raise Exception("Failed to change the working directory")
    finally:
        os.chdir(cwd)
