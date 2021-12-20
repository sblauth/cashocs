# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""This module includes finite difference Taylor tests to verify the correctness of computed gradients.

"""

from __future__ import annotations

from typing import List, Optional

import fenics
import numpy as np

from ._exceptions import InputError
from ._loggers import warning
from ._optimal_control.optimal_control_problem import OptimalControlProblem
from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem


def control_gradient_test(
    ocp: OptimalControlProblem,
    u: Optional[List[fenics.Function]] = None,
    h: Optional[List[fenics.Function]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Taylor test to verify that the computed gradient is correct for optimal control problems.

    Parameters
    ----------
    ocp : cashocs.OptimalControlProblem
        The underlying optimal control problem, for which the gradient
        of the reduced cost function shall be verified.
    u : list[fenics.Function] or None, optional
        The point, at which the gradient shall be verified. If this is ``None``,
        then the current controls of the optimization problem are used. Default is
        ``None``.
    h : list[fenics.Function] or None, optional
        The direction(s) for the directional (Gateaux) derivative. If this is ``None``,
        one random direction is chosen. Default is ``None``.
    rng : numpy.random.RandomState or None, optional
        A numpy random state for calculating a random direction

    Returns
    -------
    float
        The convergence order from the Taylor test. If this is (approximately) 2 or larger,
        everything works as expected.
    """

    initial_state = []
    for j in range(ocp.control_dim):
        temp = fenics.Function(ocp.form_handler.control_spaces[j])
        temp.vector().vec().aypx(0.0, ocp.controls[j].vector().vec())
        initial_state.append(temp)

    if u is None:
        u = []
        for j in range(ocp.control_dim):
            temp = fenics.Function(ocp.form_handler.control_spaces[j])
            temp.vector().vec().aypx(0.0, ocp.controls[j].vector().vec())
            u.append(temp)

    if not len(u) == ocp.control_dim:
        raise InputError(
            "cashocs.verification.control_gradient_test",
            "u",
            "Length of u does not match the length of controls of the problem.",
        )

    # check if u and ocp.controls coincide, if yes, make a deepcopy
    ids_u = [fun.id() for fun in u]
    ids_controls = [fun.id() for fun in ocp.controls]
    if ids_u == ids_controls:
        u = []
        for j in range(ocp.control_dim):
            temp = fenics.Function(ocp.form_handler.control_spaces[j])
            temp.vector().vec().aypx(0.0, ocp.controls[j].vector().vec())
            u.append(temp)

    if h is None:
        h = []
        for V in ocp.form_handler.control_spaces:
            temp = fenics.Function(V)
            if rng is not None:
                temp.vector()[:] = rng.rand(V.dim())
            else:
                temp.vector()[:] = np.random.rand(V.dim())
            h.append(temp)

    for j in range(ocp.control_dim):
        ocp.controls[j].vector().vec().aypx(0.0, u[j].vector().vec())

    # Compute the norm of u for scaling purposes.
    scaling = np.sqrt(ocp.form_handler.scalar_product(ocp.controls, ocp.controls))
    if scaling < 1e-3:
        scaling = 1.0

    ocp._erase_pde_memory()
    Ju = ocp.reduced_cost_functional.evaluate()
    grad_Ju = ocp.compute_gradient()
    grad_Ju_h = ocp.form_handler.scalar_product(grad_Ju, h)

    epsilons = [scaling * 1e-2 / 2 ** i for i in range(4)]
    residuals = []

    for eps in epsilons:
        for j in range(ocp.control_dim):
            ocp.controls[j].vector().vec().aypx(0.0, u[j].vector().vec())
            ocp.controls[j].vector().vec().axpy(eps, h[j].vector().vec())
        ocp._erase_pde_memory()
        Jv = ocp.reduced_cost_functional.evaluate()

        res = abs(Jv - Ju - eps * grad_Ju_h)
        residuals.append(res)

    if np.min(residuals) < 1e-14:
        warning("The Taylor remainder is close to 0, results may be inaccurate.")

    rates = compute_convergence_rates(epsilons, residuals)

    for j in range(ocp.control_dim):
        ocp.controls[j].vector().vec().aypx(0.0, initial_state[j].vector().vec())

    return np.min(rates)


def shape_gradient_test(
    sop: ShapeOptimizationProblem,
    h: Optional[fenics.Function] = None,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Taylor test to verify that the computed shape gradient is correct.

    Parameters
    ----------
    sop : cashocs.ShapeOptimizationProblem
        The underlying shape optimization problem.
    h : fenics.Function or None, optional
        The direction used to compute the directional derivative. If this is
        ``None``, then a random direction is used (default is ``None``).
    rng : numpy.random.RandomState or None, optional
        A numpy random state for calculating a random direction

    Returns
    -------
    float
        The convergence order from the Taylor test. If this is (approximately) 2 or larger,
        everything works as expected.
    """

    if h is None:
        h = fenics.Function(sop.form_handler.deformation_space)
        if rng is not None:
            h.vector()[:] = rng.rand(sop.form_handler.deformation_space.dim())
        else:
            h.vector()[:] = np.random.rand(sop.form_handler.deformation_space.dim())

    # ensure that the shape boundary conditions are applied
    [bc.apply(h.vector()) for bc in sop.form_handler.bcs_shape]

    if sop.form_handler.use_fixed_dimensions:
        h.vector()[sop.form_handler.fixed_indices] = 0.0

    transformation = fenics.Function(sop.form_handler.deformation_space)

    sop._erase_pde_memory()
    J_curr = sop.reduced_cost_functional.evaluate()
    shape_grad = sop.compute_shape_gradient()
    shape_derivative_h = sop.form_handler.scalar_product(shape_grad, h)

    box_lower = np.min(sop.mesh_handler.mesh.coordinates())
    box_upper = np.max(sop.mesh_handler.mesh.coordinates())
    length = box_upper - box_lower

    epsilons = [length * 1e-4 / 2 ** i for i in range(4)]
    residuals = []

    for idx, eps in enumerate(epsilons):
        transformation.vector().vec().aypx(0.0, h.vector().vec())
        transformation.vector().vec().scale(eps)
        if sop.mesh_handler.move_mesh(transformation):
            sop._erase_pde_memory()
            J_pert = sop.reduced_cost_functional.evaluate()

            res = abs(J_pert - J_curr - eps * shape_derivative_h)
            residuals.append(res)
            sop.mesh_handler.revert_transformation()
        else:
            warning(
                "Deformation did not yield a valid finite element mesh. Results of the test are probably not accurate."
            )
            residuals.append(float("inf"))

    if np.min(residuals) < 1e-14:
        warning("The Taylor remainder is close to 0, results may be inaccurate.")

    rates = compute_convergence_rates(epsilons, residuals)

    return np.min(rates)


def compute_convergence_rates(
    epsilons: List[float], residuals: List[float], verbose: bool = True
) -> List[float]:
    """Computes the convergence rate of the Taylor test.

    Parameters
    ----------
    epsilons : list[float]
        The step sizes
    residuals : list[float]
        The corresponding residuals
    verbose : bool, optional
        Prints the result to the console, if ``True``. Default is ``True``

    Returns
    -------
    list[float]
        The computed convergence rates
    """

    rates = []
    for i in range(1, len(epsilons)):
        rates.append(
            np.log(residuals[i] / residuals[i - 1])
            / np.log(epsilons[i] / epsilons[i - 1])
        )

    if verbose:
        print(f"Taylor test convergence rate: {rates}")

    return rates
