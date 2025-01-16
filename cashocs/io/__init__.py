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

"""Inputs and outputs."""

from cashocs.io import config
from cashocs.io import function
from cashocs.io import managers
from cashocs.io import mesh
from cashocs.io import output
from cashocs.io.config import Config
from cashocs.io.config import load_config
from cashocs.io.function import import_function
from cashocs.io.function import read_function_from_xdmf
from cashocs.io.mesh import convert
from cashocs.io.mesh import export_mesh
from cashocs.io.mesh import extract_mesh_from_xdmf
from cashocs.io.mesh import import_mesh
from cashocs.io.mesh import read_mesh_from_xdmf
from cashocs.io.mesh import write_out_mesh
from cashocs.io.output import OutputManager

__all__ = [
    "config",
    "function",
    "managers",
    "mesh",
    "output",
    "Config",
    "load_config",
    "import_function",
    "read_function_from_xdmf",
    "convert",
    "export_mesh",
    "extract_mesh_from_xdmf",
    "import_mesh",
    "read_mesh_from_xdmf",
    "write_out_mesh",
    "OutputManager",
]
