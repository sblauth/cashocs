#!/usr/bin/env python

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

"""Mesh conversion from GMSH .msh to .xdmf."""

import argparse
import json
import time
from typing import List, Optional

import meshio
import numpy as np


def _generate_parser() -> argparse.ArgumentParser:
    """Returns a parser for command line arguments."""

    parser = argparse.ArgumentParser(
        prog="cashocs-convert", description="Convert GMSH to XDMF."
    )
    parser.add_argument(
        "infile", type=str, help="GMSH file to be converted, has to end in .msh"
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="XDMF file to which the mesh shall be converted, has to end in .xdmf",
    )

    return parser


def check_file_extension(file: str, required_extension: str) -> None:
    if not file.split(".")[-1] == required_extension:
        raise Exception(
            f"Cannot use {file} due to wrong format.",
        )


def write_mesh(
    meshdim: int, points: np.ndarray, cells_dict: dict, ostring: str
) -> None:
    """Write out the main mesh with meshio.

    Args:
        meshdim: The dimension of the mesh.
        points: The array of points.
        cells_dict: The cells_dict of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.
    """

    cells_str = "triangle"
    if meshdim == 2:
        cells_str = "triangle"
    elif meshdim == 3:
        cells_str = "tetra"

    xdmf_mesh = meshio.Mesh(points=points, cells={cells_str: cells_dict[cells_str]})
    meshio.write(f"{ostring}.xdmf", xdmf_mesh)


def write_subdomains(
    meshdim: int,
    cell_data_dict: dict,
    points: np.ndarray,
    cells_dict: dict,
    ostring: str,
) -> None:
    """Write out an xdmf file with meshio corresponding to the subdomains.

    Args:
        meshdim: The dimension of the mesh.
        cell_data_dict: The cell_data_dict of the mesh.
        points: The array of points.
        cells_dict: The cells_dict of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.
    """

    cells_str = "triangle"
    if meshdim == 2:
        cells_str = "triangle"
    elif meshdim == 3:
        cells_str = "tetra"

    if "gmsh:physical" in cell_data_dict.keys():
        if cells_str in cell_data_dict["gmsh:physical"].keys():
            subdomains = meshio.Mesh(
                points=points,
                cells={cells_str: cells_dict[cells_str]},
                cell_data={"subdomains": [cell_data_dict["gmsh:physical"][cells_str]]},
            )
            meshio.write(f"{ostring}_subdomains.xdmf", subdomains)


def write_boundaries(
    meshdim: int,
    cell_data_dict: dict,
    points: np.ndarray,
    cells_dict: dict,
    ostring: str,
) -> None:
    """Write out an xdmf file with meshio corresponding to the boundaries.

    Args:
        meshdim: The dimension of the mesh.
        cell_data_dict: The cell_data_dict of the mesh.
        points: The array of points.
        cells_dict: The cells_dict of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.
    """

    facet_str = "line"
    if meshdim == 2:
        facet_str = "line"
    elif meshdim == 3:
        facet_str = "triangle"

    if "gmsh:physical" in cell_data_dict.keys():
        if facet_str in cell_data_dict["gmsh:physical"].keys():
            xdmf_boundaries = meshio.Mesh(
                points=points,
                cells={facet_str: cells_dict[facet_str]},
                cell_data={"boundaries": [cell_data_dict["gmsh:physical"][facet_str]]},
            )
            meshio.write(f"{ostring}_boundaries.xdmf", xdmf_boundaries)


def check_for_physical_names(inputfile: str, meshdim: int, ostring: str) -> None:
    """Checks and extracts physical tags if they are given as strings.

    Args:
        inputfile: Path to the input file.
        meshdim: The dimension of the mesh.
        ostring: The output string, containing the name and path to the output file,
            without extension.
    """

    physical_groups = {"dx": {}, "ds": {}}
    has_physical_groups = False
    with open(inputfile, "r") as infile:
        for line in infile:
            line = line.strip()
            if line == "$PhysicalNames":
                has_physical_groups = True
                info_line = next(infile).strip()
                no_physical_groups = int(info_line)
                for i in range(no_physical_groups):
                    physical_line = next(infile).strip().split()
                    phys_dim = int(physical_line[0])
                    phys_tag = int(physical_line[1])
                    phys_name = physical_line[2]
                    if "'" in phys_name:
                        phys_name = phys_name.replace("'", "")
                    if '"' in phys_name:
                        phys_name = phys_name.replace('"', "")

                    if phys_dim == meshdim:
                        physical_groups["dx"][phys_name] = phys_tag
                    elif phys_dim == meshdim - 1:
                        physical_groups["ds"][phys_name] = phys_tag

                break

        if has_physical_groups:
            with open(f"{ostring}_physical_groups.json", "w") as ofile:
                json.dump(physical_groups, ofile)


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
    outputfile = args.outfile
    check_file_extension(inputfile, "msh")
    check_file_extension(outputfile, "xdmf")

    ostring = outputfile.split(".")[0]

    mesh_collection = meshio.read(inputfile)

    points = mesh_collection.points
    cells_dict = mesh_collection.cells_dict
    cell_data_dict = mesh_collection.cell_data_dict

    # Check, whether we have a 2D or 3D mesh:
    keyvals = cells_dict.keys()
    meshdim = 2
    if "tetra" in keyvals:
        meshdim = 3
    elif "triangle" in keyvals:
        meshdim = 2
        points = points[:, :2]

    write_mesh(meshdim, points, cells_dict, ostring)
    write_subdomains(meshdim, cell_data_dict, points, cells_dict, ostring)
    write_boundaries(meshdim, cell_data_dict, points, cells_dict, ostring)
    check_for_physical_names(inputfile, meshdim, ostring)

    end_time = time.time()
    print(
        f"cashocs - info: Successfully converted {inputfile} to {outputfile} "
        f"in {end_time - start_time:.2f} s"
    )


if __name__ == "__main__":
    convert()
