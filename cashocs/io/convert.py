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

"""Mesh conversion directly callable from python."""

from typing import Optional

import fenics

from cashocs._cli._convert import convert as cli_convert


def convert(input_file: str, output_file: Optional[str] = None) -> None:
    """Converts the input mesh file to a xdmf mesh file for cashocs to work with.

    Args:
        input_file: A gmsh .msh file.
        output_file: The name of the output .xdmf file or ``None``. If this is ``None``,
            then a file name.msh will be converted to name.xdmf, i.e., the name of the
            input file stays the same

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        if output_file is None:
            input_name = input_file.rsplit(".", 1)[0]
            output_file = f"{input_name}.xdmf"

        cli_convert([input_file, output_file])
    fenics.MPI.barrier(fenics.MPI.comm_world)
