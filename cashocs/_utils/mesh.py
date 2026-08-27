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
    comm = mesh.mpi_comm()
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, tdim)
    mesh.init(tdim, tdim - 1)

    sub_arr = subdomains.array()
    boundary_arr = boundaries.array()
    if np.issubdtype(boundary_arr.dtype, np.unsignedinteger):
        uninitialized_boundary = np.iinfo(boundary_arr.dtype).max
    else:
        uninitialized_boundary = -1

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
    interior_ids = np.nonzero(interior)[0]
    internal_mask = cell_id_0 == cell_id_1
    uninitialized_mask = boundary_arr[interior_ids] == uninitialized_boundary
    facet_ids = interior_ids[internal_mask & uninitialized_mask]
    boundary_tags_ds = np.copy(cell_id_0[internal_mask & uninitialized_mask])

    shared_facets = np.fromiter(
        mesh.topology().shared_entities(tdim - 1).keys(),
        dtype=np.int64,
    )
    partition_facets = shared_facets[counts[shared_facets] == 1]
    partition_facets = partition_facets[
        boundary_arr[partition_facets] == uninitialized_boundary
    ]
    global_facet_ids = mesh.topology().global_indices(tdim - 1)
    local_shared_tags: dict[int, int] = {}
    if partition_facets.size > 0:
        local_shared_tags = {
            int(global_facet_ids[facet]): int(sub_arr[f2c[offsets[facet]]])
            for facet in partition_facets
        }

    shared_tags: dict[int, list[int]] = {}
    for rank_tags in comm.allgather(local_shared_tags):
        for global_facet_id, cell_tag in rank_tags.items():
            shared_tags.setdefault(global_facet_id, []).append(cell_tag)

    partition_ids = []
    partition_tags = []
    if partition_facets.size > 0:
        for facet in partition_facets:
            tags = shared_tags[int(global_facet_ids[facet])]
            if len(tags) > 1 and len(set(tags)) == 1:
                partition_ids.append(facet)
                partition_tags.append(tags[0])

    if partition_ids:
        facet_ids = np.concatenate((facet_ids, np.asarray(partition_ids)))
        boundary_tags_ds = np.concatenate(
            (boundary_tags_ds, np.asarray(partition_tags, dtype=boundary_tags_ds.dtype))
        )

    local_internal_ids = np.unique(boundary_tags_ds)
    gathered_internal_ids = comm.allgather(local_internal_ids.tolist())
    internal_ids = np.unique(
        np.concatenate(
            [
                np.asarray(ids, dtype=local_internal_ids.dtype)
                for ids in gathered_internal_ids
            ]
        )
    )

    dx_ids = np.array(list(physical_groups["dx"].values()))
    ds_ids = np.array(list(physical_groups["ds"].values()))
    max_id = np.maximum(
        dx_ids.max(initial=0),  # pylint: disable=unexpected-keyword-arg
        ds_ids.max(initial=0),  # pylint: disable=unexpected-keyword-arg
    )
    inverse_physical_groups_dx = {
        val: key for key, val in physical_groups["dx"].items()
    }

    internal_tags = {}
    internal_tag_values = {}
    for i, integer_id in enumerate(internal_ids):
        internal_tag = max_id + i + 1
        internal_tags[f"internal_{inverse_physical_groups_dx[integer_id]}"] = (
            internal_tag
        )
        internal_tag_values[int(integer_id)] = internal_tag

    boundary_tags_ds = np.asarray(
        [internal_tag_values[int(integer_id)] for integer_id in boundary_tags_ds],
        dtype=boundaries.array().dtype,
    )

    boundaries.array()[facet_ids] = boundary_tags_ds

    physical_groups["ds"].update(internal_tags)


def _update_ghost_boundaries(
    mesh: fenics.Mesh, boundaries: fenics.MeshFunction
) -> None:
    facet_dim = mesh.topology().dim() - 1
    if facet_dim == 0:
        facet_space = fenics.FunctionSpace(mesh, "CG", 1)
    else:
        facet_space = fenics.FunctionSpace(mesh, "DGT", 0)

    dofmap = facet_space.dofmap()
    n_facets = mesh.num_entities(facet_dim)
    facet_to_dof = np.fromiter(
        (dofmap.entity_dofs(mesh, facet_dim, [facet])[0] for facet in range(n_facets)),
        dtype=np.int64,
        count=n_facets,
    )

    boundary_func = fenics.Function(facet_space)
    vec = fenics.as_backend_type(boundary_func.vector()).vec()
    n_owned = boundary_func.vector().local_size()
    boundary_array = boundaries.array()
    shared_facets = np.fromiter(
        mesh.topology().shared_entities(facet_dim).keys(),
        dtype=np.int64,
    )
    if shared_facets.size == 0:
        return

    owned_facets = facet_to_dof < n_owned
    owned_dofs = facet_to_dof[owned_facets]
    owned_boundary_values = boundary_array[owned_facets]
    is_unsigned = np.issubdtype(boundary_array.dtype, np.unsignedinteger)
    if is_unsigned:
        unmarked_value = np.iinfo(boundary_array.dtype).max
        owned_values = np.full(n_owned, -1.0, dtype=float)
        marked = owned_boundary_values != unmarked_value
        owned_values[owned_dofs[marked]] = owned_boundary_values[marked]
    else:
        owned_values = np.zeros(n_owned, dtype=float)
        owned_values[owned_dofs] = owned_boundary_values

    boundary_func.vector().set_local(owned_values)
    boundary_func.vector().apply("")

    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    with vec.localForm() as local:
        synced_values = local.array_r[facet_to_dof[shared_facets]]
        if is_unsigned:
            marked = synced_values >= 0.0
            boundary_array[shared_facets[marked]] = synced_values[marked]
            boundary_array[shared_facets[~marked]] = unmarked_value
        else:
            boundary_array[shared_facets] = synced_values


def update_mesh_tags(
    mesh: fenics.Mesh,
    subdomains: fenics.MeshFunction,
    boundaries: fenics.MeshFunction,
    physical_groups: dict,
) -> None:
    """Updates the mesh for ghost values and internal facets.

    This function is always called when cashocs creates or reads in a mesh. It does the
    following:

    - Updates the mesh tags for ghosted subdomains and boundaries
    - Tags internal facets with a new integer tag and string `internal_NAME`, where
      `NAME` is the name of the corresponding subdomain.

    This way, all cell and facet entities are tagged and synchronized.

    Args:
        mesh: The computational mesh.
        subdomains: The mesh function with the cell indices.
        boundaries: The mesh function with the facet indices.
        physical_groups: The dictionary of physical groups (strings to integer).

    """
    _update_ghost_subdomains(mesh, subdomains)
    _tag_internal_facets(mesh, subdomains, boundaries, physical_groups)
    _update_ghost_boundaries(mesh, boundaries)
