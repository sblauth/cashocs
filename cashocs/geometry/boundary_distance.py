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

"""Compute the distance to the boundary."""

from __future__ import annotations

import copy
from typing import List, Optional, Union

import fenics
import numpy as np

from cashocs import _utils
from cashocs.geometry import measure


def compute_boundary_distance(
    mesh: fenics.Mesh,
    boundaries: Optional[fenics.MeshFunction] = None,
    boundary_idcs: Optional[List[Union[int, str]]] = None,
    tol: float = 1e-1,
    max_iter: int = 10,
) -> fenics.Function:
    """Computes (an approximation of) the distance to the boundary.

    The function iteratively solves the Eikonal equation to compute the distance to the
    boundary.

    The user can specify which boundaries are considered for the distance computation
    by specifying the parameters `boundaries` and `boundary_idcs`. Default is to
    consider all boundaries.

    Args:
        mesh: The dolfin mesh object, representing the computational domain
        boundaries: A meshfunction for the boundaries, which is needed in case specific
            boundaries are targeted for the distance computation (while others are
            ignored), default is `None` (all boundaries are used).
        boundary_idcs: A list of indices which indicate, which parts of the boundaries
            should be used for the distance computation, default is `None` (all
            boundaries are used).
        tol: A tolerance for the iterative solution of the eikonal equation. Default is
            1e-1.
        max_iter: Number of iterations for the iterative solution of the eikonal
            equation. Default is 10.

    Returns:
        A fenics function representing an approximation of the distance to the boundary.

    """
    function_space = fenics.FunctionSpace(mesh, "CG", 1)
    dx = measure.NamedMeasure("dx", mesh)

    ksp_options = copy.deepcopy(_utils.linalg.iterative_ksp_options)

    u = fenics.TrialFunction(function_space)
    v = fenics.TestFunction(function_space)

    u_curr = fenics.Function(function_space)
    u_prev = fenics.Function(function_space)
    norm_u_prev = fenics.sqrt(fenics.dot(fenics.grad(u_prev), fenics.grad(u_prev)))

    if (boundaries is not None) and (boundary_idcs is not None):
        if len(boundary_idcs) > 0:
            bcs = _utils.create_dirichlet_bcs(
                function_space, fenics.Constant(0.0), boundaries, boundary_idcs
            )
        else:
            bcs = fenics.DirichletBC(
                function_space,
                fenics.Constant(0.0),
                fenics.CompiledSubDomain("on_boundary"),
            )
    else:
        bcs = fenics.DirichletBC(
            function_space,
            fenics.Constant(0.0),
            fenics.CompiledSubDomain("on_boundary"),
        )

    lhs = fenics.dot(fenics.grad(u), fenics.grad(v)) * dx
    rhs = fenics.Constant(1.0) * v * dx

    _utils.assemble_and_solve_linear(
        lhs, rhs, bcs, x=u_curr.vector().vec(), ksp_options=ksp_options
    )
    u_curr.vector().apply("")

    rhs = fenics.dot(fenics.grad(u_prev) / norm_u_prev, fenics.grad(v)) * dx

    residual_form = (
        pow(
            fenics.sqrt(fenics.dot(fenics.grad(u_curr), fenics.grad(u_curr)))
            - fenics.Constant(1.0),
            2,
        )
        * dx
    )

    res_0 = np.sqrt(fenics.assemble(residual_form))

    for _ in range(max_iter):
        u_prev.vector().vec().aypx(0.0, u_curr.vector().vec())
        u_prev.vector().apply("")
        _utils.assemble_and_solve_linear(
            lhs, rhs, bcs, x=u_curr.vector().vec(), ksp_options=ksp_options
        )
        u_curr.vector().apply("")
        res = np.sqrt(fenics.assemble(residual_form))

        if res <= res_0 * tol:
            break

    return u_curr
