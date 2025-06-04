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

"""Basic mesh generation."""

from __future__ import annotations

import collections
import functools
from typing import Any, Callable, TYPE_CHECKING

import fenics
from mpi4py import MPI
import numpy as np
from typing_extensions import Literal

from cashocs import _exceptions
from cashocs import log
from cashocs import mpi
from cashocs.geometry import measure

if TYPE_CHECKING:
    from cashocs import _typing


def _get_mesh_stats(
    mode: Literal["import", "generate"],
) -> Callable[..., Callable[..., _typing.MeshTuple]]:
    """A decorator for mesh importing / generating function which logs stats.

    Args:
        mode: A string indicating whether the mesh is being generated or imported.

    Returns:
        The decorated function.

    """

    def decorator_stats(
        func: Callable[..., _typing.MeshTuple],
    ) -> Callable[..., _typing.MeshTuple]:
        """A decorator for a mesh generating function.

        Args:
            func: The function to be decorated.

        Returns:
            The decorated function

        """

        @functools.wraps(func)
        def wrapper_stats(*args: Any, **kwargs: Any) -> _typing.MeshTuple:
            """Wrapper function for mesh generating functions.

            Args:
                *args: The arguments for the function.
                **kwargs: The keyword arguments for the function.

            Returns:
                The wrapped function.

            """
            comm = None
            if "comm" in kwargs.keys():  # pylint: disable=consider-iterating-dictionary
                comm = kwargs["comm"]
            else:
                for arg in args:
                    if isinstance(arg, MPI.Comm):
                        comm = arg

            if comm is None:
                comm = mpi.COMM_WORLD

            word = "importing" if mode.casefold() == "import" else "generating"
            worded = "imported" if mode.casefold() == "import" else "generated"
            mpi_size = comm.size
            log.begin(f"{word.capitalize()} mesh.", level=log.INFO)

            value = func(*args, **kwargs)
            dim = value[0].geometry().dim()

            log.info(
                f"Successfully {worded} {dim}-dimensional mesh on {mpi_size} CPU(s)."
            )
            log.info(
                f"Mesh contains {value[0].num_entities_global(0):,} vertices and "
                f"{value[0].num_entities_global(dim):,} cells of type "
                f"{value[0].ufl_cell().cellname()}."
            )
            log.end()
            return value

        return wrapper_stats

    return decorator_stats


@_get_mesh_stats("generate")
def interval_mesh(
    n: int = 10,
    start: float = 0.0,
    end: float = 1.0,
    partitions: list[float] | None = None,
    comm: MPI.Comm | None = None,
) -> _typing.MeshTuple:
    r"""Creates an 1D interval mesh starting at x=0 to x=length.

    This function creates a uniform mesh of a 1D interval, starting at the ``start`` and
    ending at ``end``. The resulting mesh uses ``n`` sub-intervals to
    discretize the geometry. The boundary markers are as follows:

     - 1 corresponds to :math:`x=start`

     - 2 corresponds to :math:`x=end`

    Args:
        n: Number of elements for discretizing the interval, default is 10
        start: The start of the interval, default is 0.0
        end: The end of the interval, default is 1.0
        partitions: Points in the interval at which a partition in subdomains should be
            made. The resulting volume measure is sorted ascendingly according to the
            sub-intervals defined in partitions (starting at 1). Defaults to ``None``.
        comm: MPI communicator that is to be used for creating the mesh.

    Returns:
        A tuple (mesh, subdomains, boundaries, dx, ds, dS), where mesh is the imported
        FEM mesh, subdomains is a mesh function for the subdomains, boundaries is a mesh
        function for the boundaries, dx is a volume measure, ds is a surface measure,
        and dS is a measure for the interior facets.

    """
    if end <= start:
        raise _exceptions.InputError(
            "cashocs.geometry.interval_mesh", "end", "end needs to be larger than start"
        )

    if partitions is not None:
        if not all(x < y for x, y in zip(partitions, partitions[1:])):
            raise _exceptions.InputError(
                "cashocs.geometry.interval_mesh",
                "partitions",
                "partitions must be strictly increasing",
            )

    n = int(n)
    dim = 1

    if comm is None:
        comm = mpi.COMM_WORLD

    mesh = fenics.IntervalMesh(comm, n, start, end)

    subdomains = fenics.MeshFunction("size_t", mesh, dim=dim)
    boundaries = fenics.MeshFunction("size_t", mesh, dim=dim - 1)

    x_min = fenics.CompiledSubDomain(
        "on_boundary && near(x[0], start, tol)", tol=fenics.DOLFIN_EPS, start=start
    )
    x_max = fenics.CompiledSubDomain(
        "on_boundary && near(x[0], end, tol)", tol=fenics.DOLFIN_EPS, end=end
    )
    x_min.mark(boundaries, 1)
    x_max.mark(boundaries, 2)

    if partitions is not None:
        padded_partitions = collections.deque(partitions)
        padded_partitions.appendleft(start)
        padded_partitions.append(end)

        for i in range(len(padded_partitions) - 1):
            start_point = padded_partitions[i]
            end_point = padded_partitions[i + 1]

            part = fenics.CompiledSubDomain(
                "x[0] >= start_point - eps && x[0] <= end_point + eps",
                start_point=start_point,
                end_point=end_point,
                eps=fenics.DOLFIN_EPS,
            )
            part.mark(subdomains, i + 1)

    dx = measure.NamedMeasure("dx", mesh, subdomain_data=subdomains)
    ds = measure.NamedMeasure("ds", mesh, subdomain_data=boundaries)
    d_interior_facet = measure.NamedMeasure("dS", mesh)

    return mesh, subdomains, boundaries, dx, ds, d_interior_facet


@_get_mesh_stats("generate")
def regular_mesh(
    n: int = 10,
    length_x: float = 1.0,
    length_y: float = 1.0,
    length_z: float | None = None,
    diagonal: Literal["left", "right", "left/right", "right/left", "crossed"] = "right",
    comm: MPI.Comm | None = None,
) -> _typing.MeshTuple:
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
        comm: MPI communicator that is to be used for creating the mesh.

    Returns:
        A tuple (mesh, subdomains, boundaries, dx, ds, dS), where mesh is the imported
        FEM mesh, subdomains is a mesh function for the subdomains, boundaries is a mesh
        function for the boundaries, dx is a volume measure, ds is a surface measure,
        and dS is a measure for the interior facets.

    """
    if length_x <= 0.0:
        raise _exceptions.InputError(
            "cashocs.geometry.regular_mesh", "length_x", "length_x needs to be positive"
        )
    if length_y <= 0.0:
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

    if comm is None:
        comm = mpi.COMM_WORLD

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
            comm,
            fenics.Point(0, 0),
            fenics.Point(sizes),
            num_points[0],
            num_points[1],
            diagonal=diagonal,
        )
    else:
        mesh = fenics.BoxMesh(
            comm,
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

    dx = measure.NamedMeasure("dx", mesh, subdomain_data=subdomains)
    ds = measure.NamedMeasure("ds", mesh, subdomain_data=boundaries)
    d_interior_facet = measure.NamedMeasure("dS", mesh)

    return mesh, subdomains, boundaries, dx, ds, d_interior_facet


@_get_mesh_stats("generate")
def regular_box_mesh(
    n: int = 10,
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_z: float | None = None,
    end_x: float = 1.0,
    end_y: float = 1.0,
    end_z: float | None = None,
    diagonal: Literal["right", "left", "left/right", "right/left", "crossed"] = "right",
    comm: MPI.Comm | None = None,
) -> _typing.MeshTuple:
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
        comm: MPI communicator that is to be used for creating the mesh.

    Returns:
        A tuple (mesh, subdomains, boundaries, dx, ds, dS), where mesh is the imported
        FEM mesh, subdomains is a mesh function for the subdomains, boundaries is a mesh
        function for the boundaries, dx is a volume measure, ds is a surface measure,
        and dS is a measure for the interior facets.

    """
    n = int(n)

    dim = 2
    sizes = [1.0, 1.0]

    if comm is None:
        comm = mpi.COMM_WORLD

    if start_z is None and end_z is None:
        lx = end_x - start_x
        ly = end_y - start_y
        sizes = [lx, ly]
        dim = 2
    else:
        if start_z is not None and end_z is not None:
            if start_z < end_z:
                lx = end_x - start_x
                ly = end_y - start_y
                lz = end_z - start_z
                sizes = [lx, ly, lz]
                dim = 3
        else:
            raise _exceptions.InputError(
                "cashocs.geometry.regular_box_mesh",
                "start_z",
                "Incorrect input for the z-coordinate. "
                "start_z has to be smaller than end_z, "
                "or only one of them is specified.",
            )

    _check_sizes(sizes)

    size_min = np.min(sizes)
    num_points = [int(np.round(length / size_min * n)) for length in sizes]

    if start_z is None:
        mesh = fenics.RectangleMesh(
            comm,
            fenics.Point(start_x, start_y),
            fenics.Point(end_x, end_y),
            num_points[0],
            num_points[1],
            diagonal=diagonal,
        )
    else:
        mesh = fenics.BoxMesh(
            comm,
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

    dx = measure.NamedMeasure("dx", mesh, subdomain_data=subdomains)
    ds = measure.NamedMeasure("ds", mesh, subdomain_data=boundaries)
    d_interior_facet = measure.NamedMeasure("dS", mesh)

    return mesh, subdomains, boundaries, dx, ds, d_interior_facet


def _check_sizes(sizes: list[float]) -> None:
    for size in sizes:
        if size <= 0:
            raise _exceptions.InputError(
                "cashocs.geometry.regular_box_mesh",
                "start_",
                "The start values have to be smaller than the end values.",
            )
