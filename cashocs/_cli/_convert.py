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

"""Mesh conversion from GMSH .msh to .xdmf."""

import argparse
import contextlib
import json
import pathlib
import subprocess  # nosec B404
import time
from typing import Dict, List, Optional

import fenics
import meshio
import numpy as np

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils


def _generate_parser() -> argparse.ArgumentParser:
    """Returns a parser for command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cashocs-convert", description="Convert GMSH to XDMF."
    )
    parser.add_argument(
        "infile", type=str, help="GMSH file to be converted, has to end in .msh"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="XDMF file to which the mesh shall be converted, has to end in .xdmf",
        default=None,
        metavar="outfile",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        type=str,
        help="The mode used to define the subdomains and boundaries. "
        "This can be either 'physical', 'geometrical' or 'none'.",
        default="physical",
        metavar="mode",
    )

    return parser


def check_mode(mode: str) -> None:
    """Cheks, whether the supplied mode is sensible.

    Args:
        mode: The mode that should be used for the conversion.

    Returns:
        Raises an exception if the supplied mode is not supported.

    """
    if mode not in ["physical", "geometrical", "none"]:
        raise _exceptions.CashocsException(
            f"The supplied mode {mode} is invalid. "
            f"Only possible options are 'physical', 'geometrical', or 'none'."
        )


def write_mesh(
    topological_dimension: int, points: np.ndarray, cells_dict: dict, ostring: str
) -> None:
    """Write out the main mesh with meshio.

    Args:
        topological_dimension: The topological dimension of the mesh.
        points: The array of points.
        cells_dict: The cells_dict of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.

    """
    cells_str = "triangle"
    if topological_dimension == 2:
        cells_str = "triangle"
    elif topological_dimension == 3:
        cells_str = "tetra"

    xdmf_mesh = meshio.Mesh(points=points, cells={cells_str: cells_dict[cells_str]})
    meshio.write(f"{ostring}.xdmf", xdmf_mesh)


def write_subdomains(
    topological_dimension: int,
    cell_data_dict: dict,
    points: np.ndarray,
    cells_dict: dict,
    ostring: str,
    mode: str,
) -> None:
    """Writes out a xdmf file with meshio corresponding to the subdomains.

    Args:
        topological_dimension: The topological dimension of the mesh.
        cell_data_dict: The cell_data_dict of the mesh.
        points: The array of points.
        cells_dict: The cells_dict of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.
        mode: The mode which is used to define the subdomains and boundaries. Should be
            one of 'physical' (the default), 'geometrical', or 'none'.

    """
    cells_str = "triangle"
    if topological_dimension == 2:
        cells_str = "triangle"
    elif topological_dimension == 3:
        cells_str = "tetra"

    if mode == "physical":
        dict_key = "gmsh:physical"
    elif mode == "geometrical":
        dict_key = "gmsh:geometrical"
    else:
        dict_key = None

    if mode != "none" and dict_key in cell_data_dict.keys():
        if cells_str in cell_data_dict[dict_key].keys():
            subdomains = meshio.Mesh(
                points=points,
                cells={cells_str: cells_dict[cells_str]},
                cell_data={"subdomains": [cell_data_dict[dict_key][cells_str]]},
            )
            meshio.write(f"{ostring}_subdomains.xdmf", subdomains)
    else:
        if pathlib.Path(f"{ostring}_subdomains.xdmf").is_file():
            subprocess.run(  # nosec 603
                ["rm", f"{ostring}_subdomains.xdmf"], check=True
            )
        if pathlib.Path(f"{ostring}_subdomains.h5").is_file():
            subprocess.run(["rm", f"{ostring}_subdomains.h5"], check=True)  # nosec 603


def write_boundaries(
    topological_dimension: int,
    cell_data_dict: dict,
    points: np.ndarray,
    cells_dict: dict,
    ostring: str,
    mode: str,
) -> None:
    """Writes out a xdmf file with meshio corresponding to the boundaries.

    Args:
        topological_dimension: The topological dimension of the mesh.
        cell_data_dict: The cell_data_dict of the mesh.
        points: The array of points.
        cells_dict: The cells_dict of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.
        mode: The mode which is used to define the subdomains and boundaries. Should be
            one of 'physical' (the default), 'geometrical', or 'none'.

    """
    facet_str = "line"
    if topological_dimension == 2:
        facet_str = "line"
    elif topological_dimension == 3:
        facet_str = "triangle"

    if mode == "physical":
        dict_key = "gmsh:physical"
    elif mode == "geometrical":
        dict_key = "gmsh:geometrical"
    else:
        dict_key = None

    if mode != "none" and dict_key in cell_data_dict.keys():
        if facet_str in cell_data_dict[dict_key].keys():
            xdmf_boundaries = meshio.Mesh(
                points=points,
                cells={facet_str: cells_dict[facet_str]},
                cell_data={"boundaries": [cell_data_dict[dict_key][facet_str]]},
            )
            meshio.write(f"{ostring}_boundaries.xdmf", xdmf_boundaries)
    else:
        if pathlib.Path(f"{ostring}_boundaries.xdmf").is_file():
            subprocess.run(  # nosec 603
                ["rm", f"{ostring}_boundaries.xdmf"], check=True
            )
        if pathlib.Path(f"{ostring}_boundaries.h5").is_file():
            subprocess.run(["rm", f"{ostring}_boundaries.h5"], check=True)  # nosec 603


def check_for_physical_names(
    inputfile: str, topological_dimension: int, ostring: str
) -> None:
    """Checks and extracts physical tags if they are given as strings.

    Args:
        inputfile: Path to the input file.
        topological_dimension: The dimension of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.

    """
    physical_groups: Dict[str, Dict[str, int]] = {"dx": {}, "ds": {}}
    has_physical_groups = False
    with open(inputfile, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line == "$PhysicalNames":
                has_physical_groups = True
                info_line = next(infile).strip()
                no_physical_groups = int(info_line)
                for _ in range(no_physical_groups):
                    physical_line = next(infile).strip().split()
                    phys_dim = int(physical_line[0])
                    phys_tag = int(physical_line[1])
                    phys_name = physical_line[2]
                    if "'" in phys_name:
                        phys_name = phys_name.replace("'", "")
                    if '"' in phys_name:
                        phys_name = phys_name.replace('"', "")

                    if phys_dim == topological_dimension:
                        physical_groups["dx"][phys_name] = phys_tag
                    elif phys_dim == topological_dimension - 1:
                        physical_groups["ds"][phys_name] = phys_tag

                break

        if has_physical_groups:
            with open(
                f"{ostring}_physical_groups.json", "w", encoding="utf-8"
            ) as ofile:
                json.dump(physical_groups, ofile, indent=4)


def convert(argv: Optional[List[str]] = None) -> None:
    """Converts a Gmsh .msh file to a .xdmf mesh file.

    Args:
        argv: Command line options. The first parameter is the input .msh file,
            the second is the output .xdmf file

    """
    start_time = time.time()

    parser = _generate_parser()
    args = parser.parse_args(argv)

    inputfile = args.infile
    _utils.check_file_extension(inputfile, "msh")

    mode = args.mode
    check_mode(mode)

    outputfile = args.outfile
    if outputfile is None:
        outputfile = f"{inputfile[:-4]}.xdmf"
    _utils.check_file_extension(outputfile, "xdmf")

    quiet = args.quiet

    ostring = outputfile.rsplit(".", 1)[0]

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        with contextlib.redirect_stdout(None):
            mesh_collection = meshio.read(inputfile)

        points = mesh_collection.points
        cells_dict = mesh_collection.cells_dict
        cell_data_dict = mesh_collection.cell_data_dict

        # Check, whether we have a 2D or 3D mesh:
        keyvals = cells_dict.keys()
        topological_dimension = 2
        if "tetra" in keyvals:
            topological_dimension = 3
        elif "triangle" in keyvals:
            topological_dimension = 2
            # check if geometrical dimension matches topological dimension
            z_coords = points[:, 2]
            if np.abs(np.max(z_coords) - np.min(z_coords)) <= 1e-15:
                points = points[:, :2]

        write_mesh(topological_dimension, points, cells_dict, ostring)
        write_subdomains(
            topological_dimension, cell_data_dict, points, cells_dict, ostring, mode
        )
        write_boundaries(
            topological_dimension, cell_data_dict, points, cells_dict, ostring, mode
        )
        check_for_physical_names(inputfile, topological_dimension, ostring)
    fenics.MPI.barrier(fenics.MPI.comm_world)

    end_time = time.time()

    if not quiet:
        _loggers.info(
            f"Successfully converted {inputfile} to {outputfile} "
            f"in {end_time - start_time:.2f} s"
        )


if __name__ == "__main__":
    convert()
