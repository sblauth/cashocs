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

"""Inputs and outputs."""

from cashocs.io.config import Config
from cashocs.io.config import load_config
from cashocs.io.function import read_function_from_xdmf
from cashocs.io.mesh import convert
from cashocs.io.mesh import import_mesh
from cashocs.io.mesh import read_mesh_from_xdmf
from cashocs.io.mesh import write_out_mesh
from cashocs.io.output import OutputManager

__all__ = [
    "convert",
    "Config",
    "load_config",
    "write_out_mesh",
    "read_mesh_from_xdmf",
    "read_function_from_xdmf",
    "OutputManager",
    "import_mesh",
]
