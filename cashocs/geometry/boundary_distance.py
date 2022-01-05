"""
Created on 05/01/2022, 10.31

@author: blauths
"""

from __future__ import annotations

from typing import Optional, List

import fenics
import numpy as np
from petsc4py import PETSc

from .measure import _NamedMeasure
from ..utils.forms import create_dirichlet_bcs
from ..utils.linalg import (
    _setup_petsc_options,
    _solve_linear_problem,
    _assemble_petsc_system,
)


def compute_boundary_distance(
    mesh: fenics.Mesh,
    boundaries: Optional[fenics.MeshFunction] = None,
    boundary_idcs: Optional[List[int]] = None,
    tol: float = 1e-1,
    max_iter: int = 10,
) -> fenics.Function:
    """Computes (an approximation of) the distance to the boundary.

    The function iteratively solves the Eikonal equation to compute the distance to the
    boundary.

    The user can specify which boundaries are considered for the distance computation
    by specifying the parameters `boundaries` and `boundary_idcs`. Default is to
    consider all boundaries.

    Parameters
    ----------
    mesh : fenics.Mesh
        The dolfin mesh object, representing the computational domain
    boundaries : fenics.MeshFunction or None, optional
        A meshfunction for the boundaries, which is needed in case specific boundaries
        are targeted for the distance computation (while others are ignored), default
        is `None` (all boundaries are used)
    boundary_idcs : list[int] or None, optional
        A list of indices which indicate, which parts of the boundaries should be used
        for the distance computation, default is `None` (all boundaries are used).
    tol : float, optional
        A tolerance for the iterative solution of the eikonal equation. Default is 1e-1.
    max_iter : int, optional
        Number of iterations for the iterative solution of the eikonal equation. Default
        is 10.

    Returns
    -------
    fenics.Function
        A fenics function representing an approximation of the distance to the boundary.

    """

    V = fenics.FunctionSpace(mesh, "CG", 1)
    dx = _NamedMeasure("dx", mesh)

    ksp = PETSc.KSP().create()
    ksp_options = [
        ["ksp_type", "cg"],
        ["pc_type", "hypre"],
        ["pc_hypre_type", "boomeramg"],
        ["pc_hypre_boomeramg_strong_threshold", 0.7],
        ["ksp_rtol", 1e-20],
        ["ksp_atol", 1e-50],
        ["ksp_max_it", 1000],
    ]
    _setup_petsc_options([ksp], [ksp_options])

    u = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)

    u_curr = fenics.Function(V)
    u_prev = fenics.Function(V)
    norm_u_prev = fenics.sqrt(fenics.dot(fenics.grad(u_prev), fenics.grad(u_prev)))

    if (boundaries is not None) and (boundary_idcs is not None):
        if len(boundary_idcs) > 0:
            bcs = create_dirichlet_bcs(
                V, fenics.Constant(0.0), boundaries, boundary_idcs
            )
        else:
            bcs = fenics.DirichletBC(
                V, fenics.Constant(0.0), fenics.CompiledSubDomain("on_boundary")
            )
    else:
        bcs = fenics.DirichletBC(
            V, fenics.Constant(0.0), fenics.CompiledSubDomain("on_boundary")
        )

    a = fenics.dot(fenics.grad(u), fenics.grad(v)) * dx
    L = fenics.Constant(1.0) * v * dx

    A, b = _assemble_petsc_system(a, L, bcs)
    _solve_linear_problem(ksp, A, b, u_curr.vector().vec())

    L = fenics.dot(fenics.grad(u_prev) / norm_u_prev, fenics.grad(v)) * dx

    F_res = (
        pow(
            fenics.sqrt(fenics.dot(fenics.grad(u_curr), fenics.grad(u_curr)))
            - fenics.Constant(1.0),
            2,
        )
        * dx
    )

    res_0 = np.sqrt(fenics.assemble(F_res))

    for i in range(max_iter):
        u_prev.vector().vec().aypx(0.0, u_curr.vector().vec())
        A, b = _assemble_petsc_system(a, L, bcs)
        _solve_linear_problem(ksp, A, b, u_curr.vector().vec())
        res = np.sqrt(fenics.assemble(F_res))

        if res <= res_0 * tol:
            break

    return u_curr
