#!/usr/bin/env python

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

"""Mesh conversion from GMSH .msh to .xdmf."""

from __future__ import annotations

import typer

from cashocs import mpi
from cashocs.io import mesh as iomesh

app = typer.Typer(add_completion=False)


@app.command()
def convert(
    infile: str = typer.Argument(
        ..., help="GMSH file to be converted, has to end in .msh"
    ),
    outfile: str | None = typer.Option(
        None,
        "-o",
        "--outfile",
        help="XDMF output file, has to end in .xdmf. "
        "If this is not given, then the output will be the same as the input, "
        "but with .xdmf suffix.",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Whether or not to show information on stdout."
    ),
    mode: str = typer.Option(
        "physical",
        "-m",
        "--mode",
        help="The mode used to define the subdomains and boundaries. "
        "This can be either 'physical', 'geometrical' or 'none'. "
        "If the mode is 'physical', then the physical groups defined in the Gmsh file "
        "are used. "
        "If this is 'geometrical', then the geometrical groups defined in the Gmsh "
        "file are used. If this is 'none', no information is used.",
    ),
) -> None:
    """Convert a Gmsh .msh file to a .xdmf mesh file."""
    mesh_converter = iomesh.MeshConverter(mpi.COMM_WORLD)
    mesh_converter.convert(infile, outputfile=outfile, mode=mode, quiet=quiet)


if __name__ == "__main__":
    app()
