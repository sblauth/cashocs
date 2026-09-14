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

"""Extract mesh files from .xdmf files."""

from __future__ import annotations

import typer

from cashocs.io import mesh as iomesh

app = typer.Typer(add_completion=False)


@app.command()
def extract_mesh(
    xdmffile: str = typer.Argument(..., help="The XDMF file which holds the mesh."),
    iteration: int = typer.Option(
        0,
        "-i",
        "--iteration",
        help="Iteration of interest in the XDMF file.",
    ),
    outfile: str | None = typer.Option(
        None,
        "-o",
        "--outfile",
        help="Path to the output Gmsh file. If this is not specified, the file is "
        "written to the same directory as the XDMF file.",
    ),
    gmsh_file_original: str | None = typer.Option(
        None,
        "-g",
        "--gmsh_file_original",
        help="Path to the original Gmsh file used to define the mesh.",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Setting this disables verbose output."
    ),
) -> None:
    """Extract a Gmsh file from an XDMF file."""
    iomesh.extract_mesh_from_xdmf(
        xdmffile,
        iteration=iteration,
        outputfile=outfile,
        original_gmsh_file=gmsh_file_original,
        quiet=quiet,
    )


if __name__ == "__main__":
    app()
