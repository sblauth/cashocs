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

"""Mesh generation, quality, and management tools.

This module consists of tools for the fast generation of meshes into
fenics. The :py:func:`regular_mesh <cashocs.geometry.regular_mesh>` and
:py:func:`regular_box_mesh <cashocs.geometry.regular_box_mesh>` commands create 2D and
3D box meshes which are great for testing and development.
"""

from cashocs.geometry import boundary_distance
from cashocs.geometry import deformations
from cashocs.geometry import measure
from cashocs.geometry import mesh
from cashocs.geometry import mesh_handler
from cashocs.geometry import mesh_testing
from cashocs.geometry import quality
from cashocs.geometry.boundary_distance import compute_boundary_distance
from cashocs.geometry.deformations import DeformationHandler
from cashocs.geometry.measure import _EmptyMeasure
from cashocs.geometry.measure import generate_measure
from cashocs.geometry.mesh import interval_mesh
from cashocs.geometry.mesh import regular_box_mesh
from cashocs.geometry.mesh import regular_mesh
from cashocs.geometry.mesh_handler import _MeshHandler
from cashocs.geometry.quality import compute_mesh_quality
from cashocs.geometry.quality import MeshQuality

__all__ = [
    "boundary_distance",
    "deformations",
    "measure",
    "mesh",
    "mesh_handler",
    "mesh_testing",
    "quality",
    "compute_boundary_distance",
    "DeformationHandler",
    "_EmptyMeasure",
    "generate_measure",
    "interval_mesh",
    "regular_box_mesh",
    "regular_mesh",
    "_MeshHandler",
    "compute_mesh_quality",
    "MeshQuality",
]
