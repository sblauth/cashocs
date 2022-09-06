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

"""This module includes tests to verify the correctness of computed gradients."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _loggers
from cashocs import io

if TYPE_CHECKING:
    from cashocs._optimization import optimal_control
    from cashocs._optimization import shape_optimization


def _initialize_control_variable(
    ocp: optimal_control.OptimalControlProblem, u: Optional[List[fenics.Function]]
) -> List[fenics.Function]:
    """Initializes the control variable, so that it can be restored later.

    Args:
        ocp: The corresponding optimal control problem.
        u: The control variable.

    Returns:
        A copy of the control variables

    """
    if u is None:
        u = []
        for j in range(ocp.control_dim):
            temp = fenics.Function(ocp.form_handler.control_spaces[j])
            temp.vector().vec().aypx(0.0, ocp.controls[j].vector().vec())
            temp.vector().apply("")
            u.append(temp)

    # check if u and ocp.controls coincide, if yes, make a deepcopy
    ids_u = [fun.id() for fun in u]
    ids_controls = [fun.id() for fun in ocp.controls]
    if ids_u == ids_controls:
        u = []
        for j in range(ocp.control_dim):
            temp = fenics.Function(ocp.form_handler.control_spaces[j])
            temp.vector().vec().aypx(0.0, ocp.controls[j].vector().vec())
            temp.vector().apply("")
            u.append(temp)

    return u


def control_gradient_test(
    ocp: optimal_control.OptimalControlProblem,
    u: Optional[List[fenics.Function]] = None,
    h: Optional[List[fenics.Function]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Performs a Taylor test to verify that the computed gradient is correct.

    Args:
        ocp: The underlying optimal control problem, for which the gradient
            of the reduced cost function shall be verified.
        u: The point, at which the gradient shall be verified. If this is ``None``,
            then the current controls of the optimization problem are used. Default is
            ``None``.
        h: The direction(s) for the directional (Gâteaux) derivative. If this is
            ``None``, one random direction is chosen. Default is ``None``.
        rng: A numpy random state for calculating a random direction

    Returns:
        The convergence order from the Taylor test. If this is (approximately) 2 or
        larger, everything works as expected.

    """
    custom_rng = rng or np.random

    initial_state = []
    for j in range(ocp.control_dim):
        temp = fenics.Function(ocp.form_handler.control_spaces[j])
        temp.vector().vec().aypx(0.0, ocp.controls[j].vector().vec())
        temp.vector().apply("")
        initial_state.append(temp)

    u = _initialize_control_variable(ocp, u)

    if h is None:
        h = []
        for function_space in ocp.form_handler.control_spaces:
            temp = fenics.Function(function_space)
            temp.vector().vec().setValues(
                range(function_space.dim()), custom_rng.rand(function_space.dim())
            )
            temp.vector().apply("")
            h.append(temp)

    for j in range(ocp.control_dim):
        ocp.controls[j].vector().vec().aypx(0.0, u[j].vector().vec())
        ocp.controls[j].vector().apply("")

    # Compute the norm of u for scaling purposes.
    scaling = np.sqrt(ocp.form_handler.scalar_product(ocp.controls, ocp.controls))
    if scaling < 1e-3:
        scaling = 1.0

    # pylint: disable=protected-access
    ocp._erase_pde_memory()
    cost_functional_at_u = ocp.reduced_cost_functional.evaluate()
    gradient_at_u = ocp.compute_gradient()
    directional_derivative = ocp.form_handler.scalar_product(gradient_at_u, h)

    epsilons = [scaling * 1e-2 / 2**i for i in range(4)]
    residuals = []

    for eps in epsilons:
        for j in range(ocp.control_dim):
            ocp.controls[j].vector().vec().aypx(0.0, u[j].vector().vec())
            ocp.controls[j].vector().apply("")
            ocp.controls[j].vector().vec().axpy(eps, h[j].vector().vec())
            ocp.controls[j].vector().apply("")
        # pylint: disable=protected-access
        ocp._erase_pde_memory()
        cost_functional_at_v = ocp.reduced_cost_functional.evaluate()

        res = abs(
            cost_functional_at_v - cost_functional_at_u - eps * directional_derivative
        )
        residuals.append(res)

    if np.min(residuals) < 1e-14:
        _loggers.warning(
            "The Taylor remainder is close to 0, results may be inaccurate."
        )

    rates = compute_convergence_rates(epsilons, residuals)

    for j in range(ocp.control_dim):
        ocp.controls[j].vector().vec().aypx(0.0, initial_state[j].vector().vec())
        ocp.controls[j].vector().apply("")

    min_rate: float = np.min(rates)
    return min_rate


def shape_gradient_test(
    sop: shape_optimization.ShapeOptimizationProblem,
    h: Optional[List[fenics.Function]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Performs a Taylor test to verify that the computed shape gradient is correct.

    Args:
        sop: The underlying shape optimization problem.
        h: The direction used to compute the directional derivative. If this is
            ``None``, then a random direction is used (default is ``None``).
        rng: A numpy random state for calculating a random direction

    Returns:
        The convergence order from the Taylor test. If this is (approximately) 2 or
        larger, everything works as expected.

    """
    custom_rng = rng or np.random
    if h is None:
        h = [fenics.Function(sop.form_handler.deformation_space)]
        h[0].vector().set_local(custom_rng.rand(h[0].vector().local_size()))
        h[0].vector().apply("")

    # ensure that the shape boundary conditions are applied
    for bc in sop.form_handler.bcs_shape:
        bc.apply(h[0].vector())
        h[0].vector().apply("")

    if sop.form_handler.use_fixed_dimensions:
        h[0].vector().vec()[sop.form_handler.fixed_indices] = np.array(
            [0.0] * len(sop.form_handler.fixed_indices)
        )
        h[0].vector().apply("")

    transformation = fenics.Function(sop.form_handler.deformation_space)

    # pylint: disable=protected-access
    sop._erase_pde_memory()
    current_cost_functional = sop.reduced_cost_functional.evaluate()
    shape_grad = sop.compute_shape_gradient()
    shape_derivative_h = sop.form_handler.scalar_product(shape_grad, h)

    coords = io.mesh.gather_coordinates(sop.mesh_handler.mesh)
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        box_lower = np.min(coords)
        box_upper = np.max(coords)
    else:
        box_lower = 0.0
        box_upper = 0.0
    fenics.MPI.barrier(fenics.MPI.comm_world)

    box_lower = fenics.MPI.comm_world.bcast(box_lower, root=0)
    box_upper = fenics.MPI.comm_world.bcast(box_upper, root=0)
    length = box_upper - box_lower

    epsilons = [length * 1e-4 / 2**i for i in range(4)]
    residuals = []

    for eps in epsilons:
        transformation.vector().vec().aypx(0.0, h[0].vector().vec())
        transformation.vector().apply("")
        transformation.vector().vec().scale(eps)
        transformation.vector().apply("")
        if sop.mesh_handler.move_mesh(transformation):
            # pylint: disable=protected-access
            sop._erase_pde_memory()
            perturbed_cost_functional = sop.reduced_cost_functional.evaluate()

            res = abs(
                perturbed_cost_functional
                - current_cost_functional
                - eps * shape_derivative_h
            )
            residuals.append(res)
            sop.mesh_handler.revert_transformation()
        else:
            _loggers.warning(
                "Deformation did not yield a valid finite element mesh. "
                "Results of the test are probably not accurate."
            )
            residuals.append(float("inf"))

    if np.min(residuals) < 1e-14:
        _loggers.warning(
            "The Taylor remainder is close to 0, results may be inaccurate."
        )

    rates = compute_convergence_rates(epsilons, residuals)
    result: float = np.min(rates)
    return result


def compute_convergence_rates(
    epsilons: List[float], residuals: List[float], verbose: bool = True
) -> List[float]:
    """Computes the convergence rate of the Taylor test.

    Args:
        epsilons: The step sizes.
        residuals: The corresponding residuals.
        verbose: Prints the result to the console, if ``True``. Default is ``True``.

    Returns:
        The computed convergence rates

    """
    rates: List[float] = []
    for i in range(1, len(epsilons)):
        rate: float = np.log(residuals[i] / residuals[i - 1]) / np.log(
            epsilons[i] / epsilons[i - 1]
        )
        rates.append(rate)

    if verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        print(f"Taylor test convergence rate: {rates}", flush=True)
    fenics.MPI.barrier(fenics.MPI.comm_world)

    return rates
