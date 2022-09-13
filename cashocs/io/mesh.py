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

"""Module for handling mesh output."""

from __future__ import annotations

import pathlib

import fenics
import numpy as np


def create_point_representation(
    dim: int, points: np.ndarray, idcs: np.ndarray, subwrite_counter: int
) -> str:
    """Creates the representation of the mesh coordinates for gmsh .msh file.

    Args:
        dim: Dimension of the mesh.
        points: The array of the mesh coordinates.
        idcs: The list of indices of the points for the current element.
        subwrite_counter: A counter for looping over the indices.

    Returns:
        A string representation of the mesh coordinates.

    """
    mod_line = ""
    if dim == 2:
        mod_line = (
            f"{points[idcs[subwrite_counter]][0]:.16f} "
            f"{points[idcs[subwrite_counter]][1]:.16f} 0\n"
        )
    elif dim == 3:
        mod_line = (
            f"{points[idcs[subwrite_counter]][0]:.16f} "
            f"{points[idcs[subwrite_counter]][1]:.16f} "
            f"{points[idcs[subwrite_counter]][2]:.16f}\n"
        )

    return mod_line


def gather_coordinates(mesh: fenics.Mesh) -> np.ndarray:
    """Gathers the mesh coordinates on process 0 to write out the mesh to a Gmsh file.

    Args:
        mesh: The corresponding mesh.

    Returns:
        A numpy array which contains the vertex coordinates of the mesh

    """
    comm = fenics.MPI.comm_world
    rank = comm.Get_rank()
    top = mesh.topology()
    global_vertex_indices = top.global_indices(0)
    num_global_vertices = mesh.num_entities_global(0)
    local_mesh_coordinates = mesh.coordinates().copy()
    local_coordinates_list = comm.gather(local_mesh_coordinates, root=0)
    vertex_map_list = comm.gather(global_vertex_indices, root=0)

    if rank == 0:
        coordinates = np.zeros((num_global_vertices, local_mesh_coordinates.shape[1]))
        for coords, verts in zip(local_coordinates_list, vertex_map_list):
            coordinates[verts] = coords
    else:
        coordinates = np.zeros((1, 1))
    fenics.MPI.barrier(fenics.MPI.comm_world)

    return coordinates


def parse_file(
    original_msh_file: str, out_msh_file: str, points: np.ndarray, dim: int
) -> None:
    """Parses the mesh file and writes a new, corresponding one.

    Args:
        original_msh_file: Path to the original GMSH mesh file of the mesh object, has
            to end with .msh.
        out_msh_file: Path to the output mesh file, has to end with .msh.
        points: The mesh coordinates gathered on process 0
        dim: The dimensionality of the mesh

    """
    with open(original_msh_file, "r", encoding="utf-8") as old_file, open(
        out_msh_file, "w", encoding="utf-8"
    ) as new_file:
        node_section = False
        info_section = False
        subnode_counter = 0
        subwrite_counter = 0
        idcs = np.zeros(1, dtype=int)

        for line in old_file:
            if line == "$EndNodes\n":
                node_section = False

            if not node_section:
                new_file.write(line)
            else:
                split_line = line.split(" ")
                if info_section:
                    new_file.write(line)
                    info_section = False
                else:
                    if len(split_line) == 4:
                        num_subnodes = int(split_line[-1][:-1])
                        subnode_counter = 0
                        subwrite_counter = 0
                        idcs = np.zeros(num_subnodes, dtype=int)
                        new_file.write(line)
                    elif len(split_line) == 1:
                        idcs[subnode_counter] = int(split_line[0][:-1]) - 1
                        subnode_counter += 1
                        new_file.write(line)
                    elif len(split_line) == 3:
                        mod_line = create_point_representation(
                            dim, points, idcs, subwrite_counter
                        )

                        new_file.write(mod_line)
                        subwrite_counter += 1

            if line == "$Nodes\n":
                node_section = True
                info_section = True


def write_out_mesh(  # noqa: C901
    mesh: fenics.Mesh, original_msh_file: str, out_msh_file: str
) -> None:
    """Writes out mesh as Gmsh .msh file.

    This method updates the vertex positions in the ``original_gmsh_file``, the
    topology of the mesh and its connections are the same. The original GMSH
    file is kept, and a new one is generated under ``out_mesh_file``.

    Args:
        mesh: The mesh object in fenics that should be saved as Gmsh file.
        original_msh_file: Path to the original GMSH mesh file of the mesh object, has
            to end with .msh.
        out_msh_file: Path to the output mesh file, has to end with .msh.

    Notes:
        The method only works with GMSH 4.1 file format. Others might also work, but
        this is not tested or ensured in any way.

    """
    dim = mesh.geometric_dimension()

    if not pathlib.Path(out_msh_file).parent.is_dir():
        pathlib.Path(out_msh_file).parent.mkdir(parents=True, exist_ok=True)

    points = gather_coordinates(mesh)

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        parse_file(original_msh_file, out_msh_file, points, dim)
    fenics.MPI.barrier(fenics.MPI.comm_world)
