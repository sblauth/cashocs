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

"""Module for mesh importing and generation."""

from __future__ import annotations

import configparser
import json
import os
import time
from typing import Union, Optional, Tuple

import fenics
import numpy as np
from typing_extensions import Literal

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import utils
from cashocs.geometry import measure
from cashocs.geometry import mesh_quality


class Mesh(fenics.Mesh):
    """A finite element mesh."""

    def __init__(self, *args, **kwargs) -> None:
        """See base class."""

        super().__init__(*args, **kwargs)
        self._config_flag = False

    def _set_config_flag(self) -> None:
        """Indicates, that the mesh has been loaded via a config file."""

        self._config_flag = True


def _check_imported_mesh_quality(
    input_arg: Union[configparser.ConfigParser, str],
    mesh: Mesh,
    cashocs_remesh_flag: bool,
) -> None:
    """Checks the quality of an imported mesh.

    Args:
        input_arg: The argument used to import the mesh.

    Returns:

    """

    if isinstance(input_arg, configparser.ConfigParser):
        mesh_quality_tol_lower = input_arg.getfloat(
            "MeshQuality", "tol_lower", fallback=0.0
        )
        mesh_quality_tol_upper = input_arg.getfloat(
            "MeshQuality", "tol_upper", fallback=1e-15
        )

        if mesh_quality_tol_lower > 0.9 * mesh_quality_tol_upper:
            _loggers.warning(
                "You are using a lower remesh tolerance (tol_lower) close to "
                "the upper one (tol_upper). This may slow down the "
                "optimization considerably."
            )

        mesh_quality_measure = input_arg.get(
            "MeshQuality", "measure", fallback="skewness"
        )
        mesh_quality_type = input_arg.get("MeshQuality", "type", fallback="min")

        # noinspection PyTypeChecker
        current_mesh_quality = mesh_quality.compute_mesh_quality(
            mesh, mesh_quality_type, mesh_quality_measure
        )

        if not cashocs_remesh_flag:
            if current_mesh_quality < mesh_quality_tol_lower:
                raise _exceptions.InputError(
                    "cashocs.geometry.import_mesh",
                    "input_arg",
                    "The quality of the mesh file you have specified is not "
                    "sufficient for evaluating the cost functional.\n"
                    f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_lower:.3e}.",
                )

            if current_mesh_quality < mesh_quality_tol_upper:
                raise _exceptions.InputError(
                    "cashocs.geometry.import_mesh",
                    "input_arg",
                    "The quality of the mesh file you have specified is not "
                    "sufficient for computing the shape gradient.\n "
                    + f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_lower:.3e}.",
                )

        else:
            if current_mesh_quality < mesh_quality_tol_lower:
                raise _exceptions.InputError(
                    "cashocs.geometry.import_mesh",
                    "input_arg",
                    "Remeshing failed.\n"
                    "The quality of the mesh file generated through remeshing is "
                    "not sufficient for evaluating the cost functional.\n"
                    + f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_lower:.3e}.",
                )

            if current_mesh_quality < mesh_quality_tol_upper:
                raise _exceptions.InputError(
                    "cashocs.geometry.import_mesh",
                    "input_arg",
                    "Remeshing failed.\n"
                    "The quality of the mesh file generated through remeshing "
                    "is not sufficient for computing the shape gradient.\n "
                    + f"It currently is {current_mesh_quality:.3e} but has to "
                    f"be at least {mesh_quality_tol_upper:.3e}.",
                )


def import_mesh(
    input_arg: Union[str, configparser.ConfigParser]
) -> Tuple[
    Mesh,
    fenics.MeshFunction,
    fenics.MeshFunction,
    fenics.Measure,
    fenics.Measure,
    fenics.Measure,
]:
    """Imports a mesh file for use with cashocs / FEniCS.

    This function imports a mesh file that was generated by GMSH and converted to
    .xdmf with the command line function :ref:`cashocs-convert <cashocs_convert>`.
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

    start_time = time.time()
    _loggers.info("Importing mesh.")

    cashocs_remesh_flag, temp_dir = utils._parse_remesh()

    # Check for the file format
    mesh_file = None
    if isinstance(input_arg, str):
        mesh_file = input_arg
    elif isinstance(input_arg, configparser.ConfigParser):
        if not cashocs_remesh_flag:
            mesh_file = input_arg.get("Mesh", "mesh_file")
        else:
            with open(f"{temp_dir}/temp_dict.json", "r") as file:
                temp_dict = json.load(file)
            mesh_file = temp_dict["mesh_file"]

    file_string = mesh_file[:-5]

    mesh = Mesh()
    xdmf_file = fenics.XDMFFile(mesh.mpi_comm(), mesh_file)
    xdmf_file.read(mesh)
    xdmf_file.close()

    subdomains_mvc = fenics.MeshValueCollection(
        "size_t", mesh, mesh.geometric_dimension()
    )
    boundaries_mvc = fenics.MeshValueCollection(
        "size_t", mesh, mesh.geometric_dimension() - 1
    )

    if os.path.isfile(f"{file_string}_subdomains.xdmf"):
        xdmf_subdomains = fenics.XDMFFile(
            mesh.mpi_comm(), f"{file_string}_subdomains.xdmf"
        )
        xdmf_subdomains.read(subdomains_mvc, "subdomains")
        xdmf_subdomains.close()
    if os.path.isfile(f"{file_string}_boundaries.xdmf"):
        xdmf_boundaries = fenics.XDMFFile(
            mesh.mpi_comm(), f"{file_string}_boundaries.xdmf"
        )
        xdmf_boundaries.read(boundaries_mvc, "boundaries")
        xdmf_boundaries.close()

    physical_groups = None
    if os.path.isfile(f"{file_string}_physical_groups.json"):
        with open(f"{file_string}_physical_groups.json") as file:
            physical_groups = json.load(file)

    subdomains = fenics.MeshFunction("size_t", mesh, subdomains_mvc)
    boundaries = fenics.MeshFunction("size_t", mesh, boundaries_mvc)

    dx = measure._NamedMeasure(
        "dx", domain=mesh, subdomain_data=subdomains, physical_groups=physical_groups
    )
    ds = measure._NamedMeasure(
        "ds", domain=mesh, subdomain_data=boundaries, physical_groups=physical_groups
    )
    # noinspection PyPep8Naming
    dS = measure._NamedMeasure(
        "dS", domain=mesh, subdomain_data=boundaries, physical_groups=physical_groups
    )

    end_time = time.time()
    _loggers.info(f"Done importing mesh. Elapsed time: {end_time - start_time:.2f} s")
    _loggers.info(
        f"Mesh contains {mesh.num_vertices():,} vertices"
        f" and {mesh.num_cells():,} cells of type {mesh.ufl_cell().cellname()}.\n"
    )

    # Add an attribute to the mesh to show with what procedure it was generated
    mesh._set_config_flag()
    # Add the physical groups to the mesh in case they are present
    if physical_groups is not None:
        mesh._physical_groups = physical_groups

    # Check the mesh quality of the imported mesh in case a config file is passed
    _check_imported_mesh_quality(input_arg, mesh, cashocs_remesh_flag)

    return mesh, subdomains, boundaries, dx, ds, dS


def regular_mesh(
    n: int = 10,
    length_x: float = 1.0,
    length_y: float = 1.0,
    length_z: Optional[float] = None,
    diagonal: Literal["left", "right", "left/right", "right/left", "crossed"] = "right",
) -> Tuple[
    fenics.Mesh,
    fenics.MeshFunction,
    fenics.MeshFunction,
    fenics.Measure,
    fenics.Measure,
    fenics.Measure,
]:
    r"""Creates a mesh corresponding to a rectangle or cube.

    This function creates a uniform mesh of either a rectangle or a cube, starting at
    the origin and having length specified in ``length_x``, ``length_y``, and
    ``length_z``. The resulting mesh uses ``n`` elements along the shortest direction
    and accordingly many along the longer ones. The resulting domain is

    .. math::
        \begin{alignedat}{2}
        &[0, length_x] \times [0, length_y] \quad &&\text{ in } 2D, \\
        &[0, length_x] \times [0, length_y] \times [0, length_z] \quad &&\text{ in } 3D.
        \end{alignedat}

    The boundary markers are ordered as follows:

      - 1 corresponds to :math:`x=0`.

      - 2 corresponds to :math:`x=length_x`.

      - 3 corresponds to :math:`y=0`.

      - 4 corresponds to :math:`y=length_y`.

      - 5 corresponds to :math:`z=0` (only in 3D).

      - 6 corresponds to :math:`z=length_z` (only in 3D).

    Args:
        n: Number of elements in the shortest coordinate direction.
        length_x: Length in x-direction.
        length_y: Length in y-direction.
        length_z: Length in z-direction, if this is ``None``, then the geometry will be
            two-dimensional (default is ``None``).
        diagonal: This defines the type of diagonal used to create the box mesh in 2D.
            This can be one of ``"right"``, ``"left"``, ``"left/right"``,
            ``"right/left"`` or ``"crossed"``.

    Returns:
        A tuple (mesh, subdomains, boundaries, dx, ds, dS), where mesh is the imported
        FEM mesh, subdomains is a mesh function for the subdomains, boundaries is a mesh
        function for the boundaries, dx is a volume measure, ds is a surface measure,
        and dS is a measure for the interior facets.
    """

    if not length_x > 0.0:
        raise _exceptions.InputError(
            "cashocs.geometry.regular_mesh", "length_x", "length_x needs to be positive"
        )
    if not length_y > 0.0:
        raise _exceptions.InputError(
            "cashocs.geometry.regular_mesh", "length_y", "length_y needs to be positive"
        )
    if not (length_z is None or length_z > 0.0):
        raise _exceptions.InputError(
            "cashocs.geometry.regular_mesh",
            "length_z",
            "length_z needs to be positive or None (for 2D mesh)",
        )

    n = int(n)

    if length_z is None:
        sizes = [length_x, length_y]
        dim = 2
    else:
        sizes = [length_x, length_y, length_z]
        dim = 3

    size_min = np.min(sizes)
    num_points = [int(np.round(length / size_min * n)) for length in sizes]

    if length_z is None:
        mesh = fenics.RectangleMesh(
            fenics.Point(0, 0),
            fenics.Point(sizes),
            num_points[0],
            num_points[1],
            diagonal=diagonal,
        )
    else:
        mesh = fenics.BoxMesh(
            fenics.Point(0, 0, 0),
            fenics.Point(sizes),
            num_points[0],
            num_points[1],
            num_points[2],
        )

    subdomains = fenics.MeshFunction("size_t", mesh, dim=dim)
    boundaries = fenics.MeshFunction("size_t", mesh, dim=dim - 1)

    x_min = fenics.CompiledSubDomain(
        "on_boundary && near(x[0], 0, tol)", tol=fenics.DOLFIN_EPS
    )
    x_max = fenics.CompiledSubDomain(
        "on_boundary && near(x[0], length, tol)", tol=fenics.DOLFIN_EPS, length=sizes[0]
    )
    x_min.mark(boundaries, 1)
    x_max.mark(boundaries, 2)

    y_min = fenics.CompiledSubDomain(
        "on_boundary && near(x[1], 0, tol)", tol=fenics.DOLFIN_EPS
    )
    y_max = fenics.CompiledSubDomain(
        "on_boundary && near(x[1], length, tol)", tol=fenics.DOLFIN_EPS, length=sizes[1]
    )
    y_min.mark(boundaries, 3)
    y_max.mark(boundaries, 4)

    if length_z is not None:
        z_min = fenics.CompiledSubDomain(
            "on_boundary && near(x[2], 0, tol)", tol=fenics.DOLFIN_EPS
        )
        z_max = fenics.CompiledSubDomain(
            "on_boundary && near(x[2], length, tol)",
            tol=fenics.DOLFIN_EPS,
            length=sizes[2],
        )
        z_min.mark(boundaries, 5)
        z_max.mark(boundaries, 6)

    dx = measure._NamedMeasure("dx", mesh, subdomain_data=subdomains)
    ds = measure._NamedMeasure("ds", mesh, subdomain_data=boundaries)
    # noinspection PyPep8Naming
    dS = measure._NamedMeasure("dS", mesh)

    return mesh, subdomains, boundaries, dx, ds, dS


def regular_box_mesh(
    n: int = 10,
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_z: Optional[float] = None,
    end_x: float = 1.0,
    end_y: float = 1.0,
    end_z: Optional[float] = None,
    diagonal: Literal["right", "left", "left/right", "right/left", "crossed"] = "right",
) -> Tuple[
    fenics.Mesh,
    fenics.MeshFunction,
    fenics.MeshFunction,
    fenics.Measure,
    fenics.Measure,
    fenics.Measure,
]:
    r"""Creates a mesh corresponding to a rectangle or cube.

    This function creates a uniform mesh of either a rectangle
    or a cube, with specified start (``S_``) and end points (``E_``).
    The resulting mesh uses ``n`` elements along the shortest direction
    and accordingly many along the longer ones. The resulting domain is

    .. math::
        \begin{alignedat}{2}
            &[start_x, end_x] \times [start_y, end_y] \quad &&\text{ in } 2D, \\
            &[start_x, end_x] \times [start_y, end_y] \times [start_z, end_z] \quad
            &&\text{ in } 3D.
        \end{alignedat}

    The boundary markers are ordered as follows:

      - 1 corresponds to :math:`x=start_x`.

      - 2 corresponds to :math:`x=end_x`.

      - 3 corresponds to :math:`y=start_y`.

      - 4 corresponds to :math:`y=end_y`.

      - 5 corresponds to :math:`z=start_z` (only in 3D).

      - 6 corresponds to :math:`z=end_z` (only in 3D).

    Args:
        n: Number of elements in the shortest coordinate direction.
        start_x: Start of the x-interval.
        start_y: Start of the y-interval.
        start_z: Start of the z-interval, mesh is 2D if this is ``None`` (default is
            ``None``).
        end_x: End of the x-interval.
        end_y: End of the y-interval.
        end_z: End of the z-interval, mesh is 2D if this is ``None`` (default is
            ``None``).
        diagonal: This defines the type of diagonal used to create the box mesh in 2D.
            This can be one of ``"right"``, ``"left"``, ``"left/right"``,
            ``"right/left"`` or ``"crossed"``.

    Returns:
        A tuple (mesh, subdomains, boundaries, dx, ds, dS), where mesh is the imported
        FEM mesh, subdomains is a mesh function for the subdomains, boundaries is a mesh
        function for the boundaries, dx is a volume measure, ds is a surface measure,
        and dS is a measure for the interior facets.
    """

    n = int(n)

    if not start_x < end_x:
        raise _exceptions.InputError(
            "cashocs.geometry.regular_box_mesh",
            "start_x",
            "Incorrect input for the x-coordinate. "
            "start_x has to be smaller than end_x.",
        )
    if not start_y < end_y:
        raise _exceptions.InputError(
            "cashocs.geometry.regular_box_mesh",
            "start_y",
            "Incorrect input for the y-coordinate. "
            "start_y has to be smaller than end_y.",
        )
    if not ((start_z is None and end_z is None) or (start_z < end_z)):
        raise _exceptions.InputError(
            "cashocs.geometry.regular_box_mesh",
            "start_z",
            "Incorrect input for the z-coordinate. "
            "start_z has to be smaller than end_z, "
            "or only one of them is specified.",
        )

    if start_z is None:
        lx = end_x - start_x
        ly = end_y - start_y
        sizes = [lx, ly]
        dim = 2
    else:
        lx = end_x - start_x
        ly = end_y - start_y
        # noinspection PyTypeChecker
        lz = end_z - start_z
        sizes = [lx, ly, lz]
        dim = 3

    size_min = np.min(sizes)
    num_points = [int(np.round(length / size_min * n)) for length in sizes]

    if start_z is None:
        mesh = fenics.RectangleMesh(
            fenics.Point(start_x, start_y),
            fenics.Point(end_x, end_y),
            num_points[0],
            num_points[1],
            diagonal=diagonal,
        )
    else:
        mesh = fenics.BoxMesh(
            fenics.Point(start_x, start_y, start_z),
            fenics.Point(end_x, end_y, end_z),
            num_points[0],
            num_points[1],
            num_points[2],
        )

    subdomains = fenics.MeshFunction("size_t", mesh, dim=dim)
    boundaries = fenics.MeshFunction("size_t", mesh, dim=dim - 1)

    x_min = fenics.CompiledSubDomain(
        "on_boundary && near(x[0], sx, tol)", tol=fenics.DOLFIN_EPS, sx=start_x
    )
    x_max = fenics.CompiledSubDomain(
        "on_boundary && near(x[0], ex, tol)", tol=fenics.DOLFIN_EPS, ex=end_x
    )
    x_min.mark(boundaries, 1)
    x_max.mark(boundaries, 2)

    y_min = fenics.CompiledSubDomain(
        "on_boundary && near(x[1], sy, tol)", tol=fenics.DOLFIN_EPS, sy=start_y
    )
    y_max = fenics.CompiledSubDomain(
        "on_boundary && near(x[1], ey, tol)", tol=fenics.DOLFIN_EPS, ey=end_y
    )
    y_min.mark(boundaries, 3)
    y_max.mark(boundaries, 4)

    if start_z is not None:
        z_min = fenics.CompiledSubDomain(
            "on_boundary && near(x[2], sz, tol)", tol=fenics.DOLFIN_EPS, sz=start_z
        )
        z_max = fenics.CompiledSubDomain(
            "on_boundary && near(x[2], ez, tol)", tol=fenics.DOLFIN_EPS, ez=end_z
        )
        z_min.mark(boundaries, 5)
        z_max.mark(boundaries, 6)

    dx = measure._NamedMeasure("dx", mesh, subdomain_data=subdomains)
    ds = measure._NamedMeasure("ds", mesh, subdomain_data=boundaries)
    # noinspection PyPep8Naming
    dS = measure._NamedMeasure("dS", mesh)

    return mesh, subdomains, boundaries, dx, ds, dS
