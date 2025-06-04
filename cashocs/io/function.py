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

"""Function input and output."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import fenics
import h5py

from cashocs import _exceptions
from cashocs import mpi
from cashocs.io import mesh as mesh_io

if TYPE_CHECKING:
    from mpi4py import MPI


def read_function_from_xdmf(
    filename: str,
    name: str,
    family: str,
    degree: int,
    vector_dim: int = 0,
    step: int = 0,
    comm: MPI.Comm | None = None,
) -> fenics.Function:
    """Reads a function from a .xdmf file containing a checkpointed function.

    Args:
        filename: The name of the .xdmf file.
        name: The name of the function.
        family: The finite element family of the function.
        degree: The degree of the finite element.
        vector_dim: The dimension of the vector, if the function is vector-valued. In
            case that this is ``0``, a scalar finite element is assumed. Default is 0.
        step: The checkpoint number. Default is ``0``.
        comm: The MPI communicator that shall be used. Default is `None`, which means
            that COMM_WORLD is used.

    Returns:
        A fenics representation of the function stored in the file.

    """
    if comm is None:
        comm = mpi.COMM_WORLD

    mesh = mesh_io.read_mesh_from_xdmf(filename, step, comm)
    if vector_dim == 0:
        function_space = fenics.FunctionSpace(mesh, family, degree)
    else:
        function_space = fenics.VectorFunctionSpace(
            mesh, family, degree, dim=vector_dim
        )

    function = fenics.Function(function_space)
    with fenics.XDMFFile(comm, filename) as file:
        file.read_checkpoint(function, name, step)

    return function


def import_function(
    filename: str,
    function_space: fenics.FunctionSpace,
    name: str | None = None,
    step: int = 0,
) -> fenics.Function:
    """Imports a function from an xdmf file to cashocs.

    Args:
        filename: The path / filename to the xdmf file.
        function_space: The corresponding function space for the function. Note that
            this must be created on a suitable mesh, which has been reimported with
            :py:func:`cashocs.io.reimport_mesh`.
        name: The name of the function in the xdmf file.
        step: The checkpoint number. Default is ``0``.

    Returns:
        The fenics function which is stored in the xdmf file.

    """
    function = fenics.Function(function_space)
    comm = function_space.mesh().mpi_comm()

    file_path = pathlib.Path(filename)
    h5_path = file_path.with_suffix(".h5")
    if name is None:
        with h5py.File(str(h5_path), "r") as f:
            names = list(f.keys())
            if len(names) == 1:
                name = names[0]
            else:
                raise _exceptions.InputError(
                    "import_function",
                    "name",
                    "You did not specify a name for importing the function. "
                    f"Available keys are {names}.",
                )

    with fenics.XDMFFile(comm, filename) as file:
        file.read_checkpoint(function, name, step)

    return function
