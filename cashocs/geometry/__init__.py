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

"""Mesh generation and import tools.

This module consists of tools for the fast generation or import of meshes into
fenics. The :py:func:`import_mesh <cashocs.geometry.import_mesh>` function is used to
import (converted) GMSH mesh files, and the :py:func:`regular_mesh
<cashocs.geometry.regular_mesh>` and :py:func:`regular_box_mesh
<cashocs.geometry.regular_box_mesh>` commands create 2D and 3D box meshes which are
great for testing.
"""

from cashocs.geometry.boundary_distance import compute_boundary_distance
from cashocs.geometry.deformation_handler import DeformationHandler
from cashocs.geometry.measure import _EmptyMeasure
from cashocs.geometry.measure import generate_measure
from cashocs.geometry.mesh import import_mesh
from cashocs.geometry.mesh import interval_mesh
from cashocs.geometry.mesh import regular_box_mesh
from cashocs.geometry.mesh import regular_mesh
from cashocs.geometry.mesh_handler import _MeshHandler
from cashocs.geometry.mesh_quality import compute_mesh_quality
from cashocs.geometry.mesh_quality import MeshQuality

__all__ = [
    "compute_mesh_quality",
    "compute_boundary_distance",
    "DeformationHandler",
    "generate_measure",
    "_EmptyMeasure",
    "import_mesh",
    "regular_mesh",
    "interval_mesh",
    "regular_box_mesh",
    "_MeshHandler",
    "MeshQuality",
]
