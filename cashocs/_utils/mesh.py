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

"""Utilities for meshes."""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any, Literal, TYPE_CHECKING

import fenics
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from cashocs import log
from cashocs import mpi
from cashocs.geometry import measure

if TYPE_CHECKING:
    from cashocs import _typing


class CashocsMesh(fenics.Mesh):
    """A finite element mesh for use with cashocs."""

    subdomains: fenics.MeshFunction
    boundaries: fenics.MeshFunction
    dx: measure.NamedMeasure
    ds: measure.NamedMeasure
    dS: measure.NamedMeasure  # pylint: disable=invalid-name
    physical_groups: dict

    def setup_cashocs_data(
        self,
        subdomains: fenics.MeshFunction,
        boundaries: fenics.MeshFunction,
        dx: measure.NamedMeasure,
        ds: measure.NamedMeasure,
        dS: measure.NamedMeasure,  # pylint: disable=invalid-name
        physical_groups: dict,
    ) -> None:
        """Sets up the data structures for use with cashocs.

        Args:
            subdomains: The mesh tags for the subdomains.
            boundaries: The mesh tags for the boundaries.
            dx: The volume / cell measure.
            ds: The exterior surface / facet measure.
            dS: The interior surface / facet measure.
            physical_groups: The dictionary of physical groups mapping names to integer
                tags.

        """
        self.subdomains = subdomains
        self.boundaries = boundaries
        self.dx = dx
        self.ds = ds
        self.dS = dS  # pylint: disable=invalid-name
        self.physical_groups = physical_groups


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


def _update_ghost_subdomains(
    mesh: fenics.Mesh, subdomains: fenics.MeshFunction
) -> None:
    dg_space = fenics.FunctionSpace(mesh, "DG", 0)
    dofmap = dg_space.dofmap()
    n_cells = mesh.num_cells()  # incl. ghosts
    c2d = np.fromiter(
        (dofmap.cell_dofs(i)[0] for i in range(n_cells)),
        dtype=np.int64,
        count=n_cells,
    )

    sub_func = fenics.Function(dg_space)
    vec = fenics.as_backend_type(sub_func.vector()).vec()
    n_owned = sub_func.vector().local_size()

    sub_arr = subdomains.array()  # length = num_cells (incl. ghosts)

    # 1) scatter owned-cell values into the owned dof slots (vectorized)
    sub_func.vector().set_local(sub_arr[:n_owned])
    sub_func.vector().apply("")

    # 2) communicate owned -> ghost
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # 3) write ghosted values back into the MeshFunction (vectorized, all cells)
    with vec.localForm() as loc:
        sub_arr[:] = loc.array_r[c2d]


def _tag_internal_facets(
    mesh: fenics.Mesh,
    subdomains: fenics.MeshFunction,
    boundaries: fenics.MeshFunction,
    physical_groups: dict,
) -> None:
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, tdim)
    mesh.init(tdim, tdim - 1)

    sub_arr = subdomains.array()

    num_facets = mesh.num_entities(tdim - 1)
    f2c = np.array(mesh.topology()(tdim - 1, tdim)())  # flat facet->cell
    c2f = np.array(mesh.topology()(tdim, tdim - 1)())
    counts = np.bincount(c2f, minlength=num_facets)  # 1=boundary, 2=interior

    offsets = np.zeros(num_facets + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    interior = counts == 2
    start = offsets[:-1][interior]
    c0, c1 = f2c[start], f2c[start + 1]

    cell_id_0 = sub_arr[c0]  # ghost-correct now
    cell_id_1 = sub_arr[c1]
    internal_mask = cell_id_0 == cell_id_1
    boundary_tags_dx = np.copy(cell_id_0[internal_mask])
    boundary_tags_ds = np.copy(cell_id_0[internal_mask])

    dx_ids = np.array(list(physical_groups["dx"].values()))
    ds_ids = np.array(list(physical_groups["ds"].values()))
    max_id = np.maximum(
        dx_ids.max(initial=0),  # pylint: disable=unexpected-keyword-arg
        ds_ids.max(initial=0),  # pylint: disable=unexpected-keyword-arg
    )
    internal_ids = np.unique(boundary_tags_dx)

    inverse_physical_groups_dx = {
        val: key for key, val in physical_groups["dx"].items()
    }

    internal_tags = {}
    for i, integer_id in enumerate(internal_ids):
        internal_tags[f"internal_{inverse_physical_groups_dx[integer_id]}"] = (
            max_id + i + 1
        )
        boundary_tags_ds[boundary_tags_dx == integer_id] = max_id + i + 1

    facet_ids = np.nonzero(interior)[0][internal_mask]

    boundaries.array()[facet_ids] = boundary_tags_ds

    physical_groups["ds"].update(internal_tags)
