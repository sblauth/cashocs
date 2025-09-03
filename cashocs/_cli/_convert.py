#!/usr/bin/env python

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

"""Mesh conversion from GMSH .msh to .xdmf."""

from __future__ import annotations

import argparse

from cashocs import mpi
from cashocs.io import mesh as iomesh


def _generate_parser() -> argparse.ArgumentParser:
    """Returns a parser for command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cashocs-convert", description="Convert a Gmsh mesh to XDMF."
    )
    parser.add_argument(
        "infile", type=str, help="GMSH file to be converted, has to end in .msh"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="XDMF output file, has to end in .xdmf. "
        "If this is not given, then the output will be the same as the input, "
        "but with .xdmf suffix.",
        default=None,
        metavar="outfile",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Whether or not to show information on stdout.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        type=str,
        help="The mode used to define the subdomains and boundaries. "
        "This can be either 'physical', 'geometrical' or 'none'. "
        "If the mode is 'physical', then the physical groups defined in the Gmsh file "
        "are used. "
        "If this is 'geometrical', then the geometrical groups defined in the Gmsh "
        "file are used. If this is 'none', no information is used.",
        default="physical",
        metavar="mode",
    )

    return parser


def convert(argv: list[str] | None = None) -> None:
    """Converts a Gmsh .msh file to a .xdmf mesh file.

    Args:
        argv: Command line options. The first parameter is the input .msh file,
            the second is the output .xdmf file

    """
    parser = _generate_parser()
    args = parser.parse_args(argv)

    inputfile = args.infile
    outputfile = args.outfile
    mode = args.mode
    quiet = args.quiet

    mesh_converter = iomesh.MeshConverter(mpi.COMM_WORLD)
    mesh_converter.convert(inputfile, outputfile=outputfile, mode=mode, quiet=quiet)


if __name__ == "__main__":
    convert()
