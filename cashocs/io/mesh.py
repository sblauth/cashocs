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

"""Mesh input and output."""

from __future__ import annotations

import configparser
import json
import pathlib
import subprocess  # nosec B404
import sys
from types import TracebackType
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, Union

import fenics
import h5py
import numpy as np

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils
from cashocs._cli._convert import convert as cli_convert
from cashocs.geometry import measure as measure_module
from cashocs.geometry import mesh as mesh_module
from cashocs.geometry import quality
from cashocs.io import config as config_module

if TYPE_CHECKING:
    from cashocs import _typing


def _change_except_hook(config: config_module.Config) -> None:
    """Change the excepthook to delete temporary files.

    Args:
        config: The configuration file for the problem.

    """
    has_cashocs_remesh_flag, temp_dir = _utils.parse_remesh()

    if has_cashocs_remesh_flag:
        with open(f"{temp_dir}/temp_dict.json", "r", encoding="utf-8") as file:
            temp_dict: Dict = json.load(file)

        remesh_directory = temp_dict["remesh_directory"]

        def custom_except_hook(
            exctype: Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
        ) -> Any:  # pragma: no cover
            """A customized hook which is injected when an exception occurs.

            Args:
                exctype: The type of the exception.
                value: The value of the exception.
                traceback: The traceback of the exception.

            """
            _loggers.debug(
                "An exception was raised by cashocs, "
                "deleting the created temporary files."
            )
            if (
                not config.getboolean("Debug", "remeshing")
                and fenics.MPI.rank(fenics.MPI.comm_world) == 0
            ):
                assert temp_dir is not None  # nosec B101
                subprocess.run(["rm", "-r", temp_dir], check=False)  # nosec B603, B607
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", remesh_directory], check=False
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)
            sys.__excepthook__(exctype, value, traceback)

        sys.excepthook = custom_except_hook  # type: ignore


def _check_imported_mesh_quality(
    input_arg: Union[str, config_module.Config],
    mesh: mesh_module.Mesh,
    cashocs_remesh_flag: bool,
) -> None:
    """Checks the quality of an imported mesh.

    This function raises exceptions when the mesh does not satisfy the desired quality
    criteria.

    Args:
        input_arg: The argument used to import the mesh.
        mesh: The finite element mesh whose quality shall be checked.
        cashocs_remesh_flag: A flag, indicating whether remeshing is active.

    """
    if isinstance(input_arg, configparser.ConfigParser):
        mesh_quality_tol_lower = input_arg.getfloat("MeshQuality", "tol_lower")
        mesh_quality_tol_upper = input_arg.getfloat("MeshQuality", "tol_upper")

        if mesh_quality_tol_lower > 0.9 * mesh_quality_tol_upper:
            _loggers.warning(
                "You are using a lower remesh tolerance (tol_lower) close to "
                "the upper one (tol_upper). This may slow down the "
                "optimization considerably."
            )

        mesh_quality_measure = input_arg.get("MeshQuality", "measure")
        mesh_quality_type = input_arg.get("MeshQuality", "type")

        current_mesh_quality = quality.compute_mesh_quality(
            mesh, mesh_quality_type, mesh_quality_measure
        )

        failed = False
        fail_msg = None
        if not cashocs_remesh_flag:
            if current_mesh_quality < mesh_quality_tol_lower:
                failed = True
                fail_msg = (
                    "The quality of the mesh file you have specified is not "
                    "sufficient for evaluating the cost functional.\n"
                    f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_lower:.3e}."
                )

            if current_mesh_quality < mesh_quality_tol_upper:
                failed = True
                fail_msg = (
                    "The quality of the mesh file you have specified is not "
                    "sufficient for computing the shape gradient.\n "
                    + f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_lower:.3e}."
                )

        else:
            if current_mesh_quality < mesh_quality_tol_lower:
                failed = True
                fail_msg = (
                    "Remeshing failed.\n"
                    "The quality of the mesh file generated through remeshing is "
                    "not sufficient for evaluating the cost functional.\n"
                    + f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_lower:.3e}."
                )

            if current_mesh_quality < mesh_quality_tol_upper:
                failed = True
                fail_msg = (
                    "Remeshing failed.\n"
                    "The quality of the mesh file generated through remeshing "
                    "is not sufficient for computing the shape gradient.\n "
                    + f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_upper:.3e}."
                )
        if failed:
            raise _exceptions.InputError(
                "cashocs.geometry.import_mesh", "input_arg", fail_msg
            )


@mesh_module._get_mesh_stats(mode="import")  # pylint:disable=protected-access
def import_mesh(input_arg: Union[str, config_module.Config]) -> _typing.MeshTuple:
    """Imports a mesh file for use with cashocs / FEniCS.

    This function imports a mesh file that was generated by GMSH and converted to
    .xdmf with the function :py:func:`cashocs.convert`.
    If there are Physical quantities specified in the GMSH file, these are imported
    to the subdomains and boundaries output of this function and can also be directly
    accessed via the measures, e.g., with ``dx(1)``, ``ds(1)``, etc.

    Args:
        input_arg: This is either a string, in which case it corresponds to the location
            of the mesh file in .xdmf file format, or a config file that
            has this path stored in its settings, under the section Mesh, as
            parameter ``mesh_file``.

    Returns:
        A tuple (mesh, subdomains, boundaries, dx, ds, dS), where mesh is the imported
        FEM mesh, subdomains is a mesh function for the subdomains, boundaries is a mesh
        function for the boundaries, dx is a volume measure, ds is a surface measure,
        and dS is a measure for the interior facets.

    Notes:
        In case the boundaries in the Gmsh .msh file are not only marked with numbers
        (as physical groups), but also with names (i.e. strings), these strings can be
        used with the integration measures ``dx`` and ``ds`` returned by this method.
        E.g., if one specified the following in a 2D Gmsh .geo file ::

            Physical Surface("domain", 1) = {i,j,k};

        where i,j,k are representative for some integers, then this can be used in the
        measure ``dx`` (as we are 2D) as follows. The command ::

            dx(1)

        is completely equivalent to ::

           dx("domain")

        and both can be used interchangeably.

    """
    cashocs_remesh_flag, temp_dir = _utils.parse_remesh()

    if not isinstance(input_arg, str):
        _change_except_hook(input_arg)

    # Check for the file format
    mesh_file: str = ""
    if isinstance(input_arg, str):
        mesh_file = input_arg
    elif isinstance(input_arg, configparser.ConfigParser):
        if not cashocs_remesh_flag:
            mesh_file = input_arg.get("Mesh", "mesh_file")
        else:
            with open(f"{temp_dir}/temp_dict.json", "r", encoding="utf-8") as file:
                temp_dict: Dict = json.load(file)
            mesh_file = temp_dict["mesh_file"]

    file_string = mesh_file[:-5]

    mesh = mesh_module.Mesh()
    xdmf_file = fenics.XDMFFile(mesh.mpi_comm(), mesh_file)
    xdmf_file.read(mesh)
    xdmf_file.close()

    subdomains_mvc = fenics.MeshValueCollection(
        "size_t", mesh, mesh.geometric_dimension()
    )
    boundaries_mvc = fenics.MeshValueCollection(
        "size_t", mesh, mesh.geometric_dimension() - 1
    )

    subdomains_path = pathlib.Path(f"{file_string}_subdomains.xdmf")
    if subdomains_path.is_file():
        xdmf_subdomains = fenics.XDMFFile(mesh.mpi_comm(), str(subdomains_path))
        xdmf_subdomains.read(subdomains_mvc, "subdomains")
        xdmf_subdomains.close()

    boundaries_path = pathlib.Path(f"{file_string}_boundaries.xdmf")
    if boundaries_path.is_file():
        xdmf_boundaries = fenics.XDMFFile(mesh.mpi_comm(), str(boundaries_path))
        xdmf_boundaries.read(boundaries_mvc, "boundaries")
        xdmf_boundaries.close()

    physical_groups: Optional[Dict[str, Dict[str, int]]] = None
    physical_groups_path = pathlib.Path(f"{file_string}_physical_groups.json")
    if physical_groups_path.is_file():
        with physical_groups_path.open("r", encoding="utf-8") as file:
            physical_groups = json.load(file)

    subdomains = fenics.MeshFunction("size_t", mesh, subdomains_mvc)
    boundaries = fenics.MeshFunction("size_t", mesh, boundaries_mvc)

    dx = measure_module.NamedMeasure(
        "dx", domain=mesh, subdomain_data=subdomains, physical_groups=physical_groups
    )
    ds = measure_module.NamedMeasure(
        "ds", domain=mesh, subdomain_data=boundaries, physical_groups=physical_groups
    )
    d_interior_facet = measure_module.NamedMeasure(
        "dS", domain=mesh, subdomain_data=boundaries, physical_groups=physical_groups
    )

    # Add an attribute to the mesh to show with what procedure it was generated
    # pylint: disable=protected-access
    mesh._set_config_flag()
    # Add the physical groups to the mesh in case they are present
    if physical_groups is not None:
        mesh.physical_groups = physical_groups

    # Check the mesh quality of the imported mesh in case a config file is passed
    _check_imported_mesh_quality(input_arg, mesh, cashocs_remesh_flag)

    return mesh, subdomains, boundaries, dx, ds, d_interior_facet


def convert(
    input_file: str, output_file: Optional[str] = None, quiet: bool = False
) -> None:
    """Converts the input mesh file to a xdmf mesh file for cashocs to work with.

    Args:
        input_file: A gmsh .msh file.
        output_file: The name of the output .xdmf file or ``None``. If this is ``None``,
            then a file name.msh will be converted to name.xdmf, i.e., the name of the
            input file stays the same
        quiet: A boolean flag which silences the output.

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        args = [input_file]

        if output_file is not None:
            args += ["-o", output_file]
        if quiet:
            args += ["-q"]

        cli_convert(args)

    fenics.MPI.barrier(fenics.MPI.comm_world)


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


def read_mesh_from_xdmf(filename: str, step: int = 0) -> mesh_module.Mesh:
    """Reads a mesh from a .xdmf file containing a checkpointed function.

    Args:
        filename: The filename to the .xdmf file.
        step: The checkpoint number. Default is ``0``.

    Returns:
        The corresponding mesh for the checkpoint number.

    """
    h5_filename = f"{filename[:-5]}.h5"
    with h5py.File(h5_filename) as file:
        name = list(file.keys())[0]
        step_name = f"{name}_{step}"
        coordinates = file[name][step_name]["mesh"]["geometry"][()]
        cells = file[name][step_name]["mesh"]["topology"][()]

    gdim = coordinates.shape[1]

    if cells.shape[1] == 2:
        tdim = 1
        cell_type = "line"
    elif cells.shape[1] == 3:
        tdim = 2
        cell_type = "triangle"
    elif cells.shape[2] == 4:
        tdim = 3
        cell_type = "tetrahedron"
    else:
        raise _exceptions.CashocsException("The mesh saved in the xdmf file is faulty.")

    if tdim > gdim:
        raise _exceptions.CashocsException(
            "The topological dimension of a mesh must not be larger than its "
            "geometrical dimension"
        )

    mesh = mesh_module.Mesh()

    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        mesh_editor = fenics.MeshEditor()
        mesh_editor.open(mesh, cell_type, tdim, gdim)
        mesh_editor.init_vertices(coordinates.shape[0])
        mesh_editor.init_cells(cells.shape[0])

        for i, vertex in enumerate(coordinates):
            mesh_editor.add_vertex(i, vertex)
        for i, cell in enumerate(cells):
            mesh_editor.add_cell(i, cell)

        mesh_editor.close()
        mesh.init()
        mesh.order()

    fenics.MeshPartitioning.build_distributed_mesh(mesh)

    return mesh
