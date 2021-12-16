#!/usr/bin/env python

# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Mesh conversion from GMSH .msh to .xdmf.

"""

import argparse
import json
import sys
import time
from typing import List, Optional

import meshio


def _generate_parser() -> argparse.ArgumentParser:
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


def convert(argv: Optional[List[str]] = None) -> None:
    """Converts a Gmsh .msh file to a .xdmf mesh file

    Parameters
    ----------
    argv : list[str]
        Command line options. The first parameter is the input .msh file,
        the second is the output .xdmf file

    Returns
    -------
    None

    """

    start_time = time.time()

    parser = _generate_parser()
    args = parser.parse_args(argv)

    inputfile = args.infile
    outputfile = args.outfile
    # Check that the inputfile has .msh file format
    if not (inputfile[-4:] == ".msh"):
        print("Error: Cannot use the current file format as input.")
        sys.exit(2)

    # Check that the outputfile has .xdmf format
    if outputfile[-5:] == ".xdmf":
        ostring = outputfile[:-5]
    else:
        print("Error: Cannot use the current file format as output.")
        sys.exit(2)

    mesh_collection = meshio.read(inputfile)

    points = mesh_collection.points
    cells_dict = mesh_collection.cells_dict
    cell_data_dict = mesh_collection.cell_data_dict

    # Check, whether we have a 2D or 3D mesh:
    keyvals = cells_dict.keys()
    if "tetra" in keyvals:
        meshdim = 3
    elif "triangle" in keyvals:
        meshdim = 2
    else:
        print("Error: This is not a valid input mesh.")
        sys.exit(2)

    if meshdim == 2:
        points = points[:, :2]
        xdmf_mesh = meshio.Mesh(
            points=points, cells={"triangle": cells_dict["triangle"]}
        )
        meshio.write(f"{ostring}.xdmf", xdmf_mesh)

        if "gmsh:physical" in cell_data_dict.keys():
            if "triangle" in cell_data_dict["gmsh:physical"].keys():
                subdomains = meshio.Mesh(
                    points=points,
                    cells={"triangle": cells_dict["triangle"]},
                    cell_data={
                        "subdomains": [cell_data_dict["gmsh:physical"]["triangle"]]
                    },
                )
                meshio.write(f"{ostring}_subdomains.xdmf", subdomains)

            if "line" in cell_data_dict["gmsh:physical"].keys():
                xdmf_boundaries = meshio.Mesh(
                    points=points,
                    cells={"line": cells_dict["line"]},
                    cell_data={"boundaries": [cell_data_dict["gmsh:physical"]["line"]]},
                )
                meshio.write(f"{ostring}_boundaries.xdmf", xdmf_boundaries)

    elif meshdim == 3:
        xdmf_mesh = meshio.Mesh(points=points, cells={"tetra": cells_dict["tetra"]})
        meshio.write(f"{ostring}.xdmf", xdmf_mesh)

        if "gmsh:physical" in cell_data_dict.keys():
            if "tetra" in cell_data_dict["gmsh:physical"].keys():
                subdomains = meshio.Mesh(
                    points=points,
                    cells={"tetra": cells_dict["tetra"]},
                    cell_data={
                        "subdomains": [cell_data_dict["gmsh:physical"]["tetra"]]
                    },
                )
                meshio.write(f"{ostring}_subdomains.xdmf", subdomains)

            if "triangle" in cell_data_dict["gmsh:physical"].keys():
                xdmf_boundaries = meshio.Mesh(
                    points=points,
                    cells={"triangle": cells_dict["triangle"]},
                    cell_data={
                        "boundaries": [cell_data_dict["gmsh:physical"]["triangle"]]
                    },
                )
                meshio.write(f"{ostring}_boundaries.xdmf", xdmf_boundaries)

    # Check for physical names
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

    end_time = time.time()
    print(
        f"cashocs - info: Successfully converted {inputfile} to {outputfile} in {end_time - start_time:.2f} s"
    )


if __name__ == "__main__":
    convert()
