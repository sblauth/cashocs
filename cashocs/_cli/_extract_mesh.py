#!/usr/bin/env python

# Copyright (C) 2020-2024 Sebastian Blauth
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

import argparse
from typing import List, Optional

from cashocs.io import mesh as iomesh


def _generate_parser() -> argparse.ArgumentParser:
    """Returns a parser for command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cashocs-extract_mesh", description="Extract a Gmsh file from an XDMF file"
    )
    parser.add_argument(
        "xdmffile", type=str, help="The base XDMF state file, which holds the mesh"
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        help="Iteration of interest in the XDMF state file. Defaults to 0.",
        default=0,
        metavar="iteration",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to the output Gmsh file. When this is not specified, the file is "
        "written to the same directory as the XDMF state file.",
        default=None,
        metavar="outfile",
    )
    parser.add_argument(
        "-g",
        "--gmsh_file_original",
        type=str,
        help="Path to the original Gmsh file used to define the mesh.",
        default=None,
        metavar="gmsh_file_original",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Setting this disables verbose output.",
    )

    return parser


def extract_mesh(argv: Optional[List[str]] = None) -> None:
    """Wrapper for calling :py:func:`cashocs.io.extract_mesh` from command line.

    Args:
        argv: The command line arguments.

    """
    parser = _generate_parser()
    args = parser.parse_args(argv)

    xdmffile = args.xdmffile
    iteration = args.iteration
    outputfile = args.outfile
    gmsh_file_original = args.gmsh_file_original
    quiet = args.quiet

    iomesh.extract_mesh_from_xdmf(
        xdmffile,
        iteration=iteration,
        outputfile=outputfile,
        original_gmsh_file=gmsh_file_original,
        quiet=quiet,
    )


if __name__ == "__main__":
    extract_mesh()
