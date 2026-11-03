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
import numpy as np
import pytest

import cashocs


@pytest.mark.parametrize("provide_space", [False, True])
def test_create_material_parameter(mesh_dir, provide_space):
    mesh, _, _, dx, _, _ = cashocs.import_mesh(str(mesh_dir / "square" / "square.xdmf"))

    material_dict = {
        ("top_left", "top_right"): 1.0,
        "bottom_left": 2.0,
        "bottom_right": 3.4,
    }

    V = FunctionSpace(mesh, "DG", 0) if provide_space else None
    parameter = cashocs.create_material_parameter(material_dict, mesh, V)

    error = assemble((parameter - 1.0) ** 2 * dx(["top_left", "top_right"]))
    error += assemble((parameter - 2.0) ** 2 * dx("bottom_left"))
    error += assemble((parameter - 3.4) ** 2 * dx("bottom_right"))
    error = np.sqrt(error)

    assert error == pytest.approx(0, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize("provide_space", [False, True])
def test_create_boundary_parameter(mesh_dir, provide_space):
    mesh, _, _, _, ds, _ = cashocs.import_mesh(str(mesh_dir / "square" / "square.xdmf"))

    boundary_dict = {
        ("left", "right"): 1.0,
        "bottom": 2.0,
        8: 3.4,
    }
    space = FunctionSpace(mesh, "DGT", 0) if provide_space else None
    parameter = cashocs.create_boundary_parameter(boundary_dict, mesh, space)

    error = assemble((parameter - 1.0) ** 2 * ds(["left", "right"]))
    error += assemble((parameter - 2.0) ** 2 * ds("bottom"))
    error += assemble((parameter - 3.4) ** 2 * ds("top"))

    assert np.sqrt(error) == pytest.approx(0, rel=1e-15, abs=1e-15)
