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

"""Compute the distance to the boundary."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import fenics

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl
import numpy as np

from cashocs import _exceptions
from cashocs import _utils
from cashocs import nonlinear_solvers
from cashocs.geometry import measure

if TYPE_CHECKING:
    from cashocs import _typing


def compute_boundary_distance(
    mesh: fenics.Mesh,
    boundaries: fenics.MeshFunction | None = None,
    boundary_idcs: list[int | str] | None = None,
    tol: float = 1e-1,
    max_iter: int = 10,
    minimum_distance: float = 0.0,
    method: str = "eikonal",
) -> fenics.Function:
    """Computes (an approximation of) the distance to the boundary.

    The user can specify which boundaries are considered for the distance computation
    by specifying the parameters `boundaries` and `boundary_idcs`. Default is to
    consider all boundaries.

    Args:
        mesh: The dolfin mesh object, representing the computational domain.
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
        minimum_distance: The distance of the mesh boundary to the (physical) wall. This
            should be set to 0.0 for most applications, which is also the default. One
            exception is for turbulence modeling and wall functions.
        method: Which method should be used to compute the boundary distance. Can either
            be 'poisson' or 'eikonal'. The default is 'eikonal', which is more accurate.
            The poisson approach is less accurate, but more robust and gives a very
            good approximation close to the wall.

    Returns:
        A fenics function representing an approximation of the distance to the boundary.

    """
    if method == "poisson":
        return compute_boundary_distance_poisson(
            mesh,
            boundaries=boundaries,
            boundary_idcs=boundary_idcs,
            minimum_distance=minimum_distance,
        )
    elif method == "eikonal":
        return compute_boundary_distance_eikonal(
            mesh,
            boundaries=boundaries,
            boundary_idcs=boundary_idcs,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        raise _exceptions.InputError(
            "compute_boundary_distance",
            "method",
            "The method can only be 'poisson' or 'eikonal'.",
        )


def compute_boundary_distance_poisson(
    mesh: fenics.Mesh,
    boundaries: fenics.MeshFunction | None = None,
    boundary_idcs: list[int | str] | None = None,
    minimum_distance: float = 0.0,
) -> fenics.Function:
    """Computes the distance to the boundary with a Poisson approach.

    The user can specify which boundaries are considered for the distance computation
    by specifying the parameters `boundaries` and `boundary_idcs`. Default is to
    consider all boundaries.

    The approach is described, e.g., in Section 1.1.3 of
    `Tucker, Differential equation-based wall distance computation for DES and RANS
    (2003) <https://doi.org/10.1016/S0021-9991(03)00272-9>`_.

    Args:
        mesh (fenics.Mesh): The dolfin mesh object, representing the computational
            domain
        boundaries (fenics.MeshFunction | None, optional): A meshfunction for the
            boundaries, which is needed in case specific boundaries are targeted for
            the distance computation (while others are ignored), default is `None` (all
            boundaries are used). Defaults to None.
        boundary_idcs (list[int  |  str] | None, optional): A list of indices which
            indicate, which parts of the boundaries should be used for the distance
            computation, default is `None` (all boundaries are used). Defaults to None.
        minimum_distance (float, optional): The distance of the mesh boundary to the
            (physical) wall. This should be set to 0.0 for most applications, which is
            also the default. One exception is for turbulence modeling and wall
            functions. Defaults to 0.0.

    Returns:
        A fenics function representing an approximation of the distance to the boundary.

    """
    cg1_space = fenics.FunctionSpace(mesh, "CG", 1)
    dx = ufl.Measure("dx", domain=mesh)

    u = fenics.Function(cg1_space)
    v = fenics.TestFunction(cg1_space)
    poisson_form = (
        ufl.dot(ufl.grad(u), ufl.grad(v)) * dx - fenics.Constant(1.0) * v * dx
    )

    if (boundaries is not None) and (boundary_idcs is not None):
        if len(boundary_idcs) > 0:
            bcs = _utils.create_dirichlet_bcs(
                cg1_space, fenics.Constant(minimum_distance), boundaries, boundary_idcs
            )
        else:
            bcs = fenics.DirichletBC(
                cg1_space,
                fenics.Constant(minimum_distance),
                fenics.CompiledSubDomain("on_boundary"),
            )
    else:
        bcs = fenics.DirichletBC(
            cg1_space,
            fenics.Constant(minimum_distance),
            fenics.CompiledSubDomain("on_boundary"),
        )

    ksp_options: _typing.KspOption = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_strong_threshold": 0.7,
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-50,
        "ksp_max_it": 1000,
    }
    nonlinear_solvers.linear_solve(poisson_form, u, bcs, ksp_options=ksp_options)

    distance = fenics.Function(cg1_space)
    form = (
        distance * v * dx
        - ufl.sqrt(ufl.dot(ufl.grad(u), ufl.grad(u)) + fenics.Constant(2.0) * u)
        * v
        * dx
        + ufl.sqrt(ufl.dot(ufl.grad(u), ufl.grad(u))) * v * dx
    )
    nonlinear_solvers.linear_solve(form, distance, bcs, ksp_options=ksp_options)

    return distance


def compute_boundary_distance_eikonal(
    mesh: fenics.Mesh,
    boundaries: fenics.MeshFunction | None = None,
    boundary_idcs: list[int | str] | None = None,
    tol: float = 1e-1,
    max_iter: int = 10,
) -> fenics.Function:
    """Computes the distance to the boundary by solving the Eikonal equation.

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
    norm_u_prev = ufl.sqrt(ufl.dot(ufl.grad(u_prev), ufl.grad(u_prev)))

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

    lhs = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    rhs = fenics.Constant(1.0) * v * dx

    _utils.assemble_and_solve_linear(lhs, rhs, u_curr, bcs=bcs, ksp_options=ksp_options)

    rhs = ufl.dot(ufl.grad(u_prev) / norm_u_prev, ufl.grad(v)) * dx

    residual_form = (
        pow(
            ufl.sqrt(ufl.dot(ufl.grad(u_curr), ufl.grad(u_curr)))
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
            lhs, rhs, u_curr, bcs=bcs, ksp_options=ksp_options
        )
        res = np.sqrt(fenics.assemble(residual_form))

        if res <= res_0 * tol:
            break

    return u_curr
