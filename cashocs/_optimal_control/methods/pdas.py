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

"""A primal dual active set strategy for control constraints.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import numpy as np

from .pdas_inner_solvers import InnerNCG, InnerGradientDescent, InnerLBFGS, InnerNewton
from ..._optimal_control import ControlOptimizationAlgorithm


if TYPE_CHECKING:
    from ..optimal_control_problem import OptimalControlProblem


class PDAS(ControlOptimizationAlgorithm):
    """A primal-dual-active-set method."""

    def __init__(self, optimization_problem: OptimalControlProblem) -> None:
        """
        Parameters
        ----------
        optimization_problem : OptimalControlProblem
            The OptimalControlProblem object
        """

        super().__init__(optimization_problem)

        self.idx_active_upper_prev = [
            np.array([]) for j in range(optimization_problem.control_dim)
        ]
        self.idx_active_lower_prev = [
            np.array([]) for j in range(optimization_problem.control_dim)
        ]
        self.initialized = False
        self.mu = [
            fenics.Function(optimization_problem.control_spaces[j])
            for j in range(optimization_problem.control_dim)
        ]
        self.shift_mult = self.config.getfloat(
            "AlgoPDAS", "pdas_regularization_parameter"
        )
        self.verbose = self.config.getboolean("Output", "verbose", fallback=True)

        self.control_constraints = optimization_problem.control_constraints

        self.inner_pdas = self.config.get("AlgoPDAS", "inner_pdas")
        if self.inner_pdas in ["gradient_descent", "gd"]:
            self.inner_solver = InnerGradientDescent(optimization_problem)
        elif self.inner_pdas in ["cg", "conjugate_gradient", "ncg", "nonlinear_cg"]:
            self.inner_solver = InnerNCG(optimization_problem)
        elif self.inner_pdas in ["lbfgs", "bfgs"]:
            self.inner_solver = InnerLBFGS(optimization_problem)
        elif self.inner_pdas == "newton":
            self.inner_solver = InnerNewton(optimization_problem)

    def compute_active_inactive_sets(self) -> None:
        """Computes the active and inactive sets.

        This implementation differs slightly from the one in
        cashocs._forms.ControlFormHandler as it is needed for the PDAS.

        Returns
        -------
        None
        """

        self.idx_active_lower = [
            (
                self.mu[j].vector()[:]
                + self.shift_mult
                * (
                    self.controls[j].vector()[:]
                    - self.control_constraints[j][0].vector()[:]
                )
                < 0
            ).nonzero()[0]
            for j in range(self.form_handler.control_dim)
        ]
        self.idx_active_upper = [
            (
                self.mu[j].vector()[:]
                + self.shift_mult
                * (
                    self.controls[j].vector()[:]
                    - self.control_constraints[j][1].vector()[:]
                )
                > 0
            ).nonzero()[0]
            for j in range(self.form_handler.state_dim)
        ]

        self.idx_active = [
            np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j]))
            for j in range(self.form_handler.control_dim)
        ]
        [self.idx_active[j].sort() for j in range(self.form_handler.control_dim)]

        self.idx_inactive = [
            np.setdiff1d(
                np.arange(self.form_handler.control_spaces[j].dim()), self.idx_active[j]
            )
            for j in range(self.form_handler.control_dim)
        ]

        if self.initialized:
            if all(
                [
                    np.array_equal(
                        self.idx_active_upper[j], self.idx_active_upper_prev[j]
                    )
                    and np.array_equal(
                        self.idx_active_lower[j], self.idx_active_lower_prev[j]
                    )
                    for j in range(self.form_handler.control_dim)
                ]
            ):
                self.converged = True

        self.idx_active_upper_prev = [
            self.idx_active_upper[j] for j in range(self.form_handler.control_dim)
        ]
        self.idx_active_lower_prev = [
            self.idx_active_lower[j] for j in range(self.form_handler.control_dim)
        ]
        self.initialized = True

    def run(self) -> None:
        """Solves the optimization problem with the primal-dual-active-set method.

        Returns
        -------
        None
        """

        self.iteration = 0

        ### TODO: Check for feasible initialization

        self.compute_active_inactive_sets()

        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False
        self.objective_value = self.cost_functional.evaluate()
        self.state_problem.has_solution = True
        self.gradient_problem.solve()
        norm_init = np.sqrt(self._stationary_measure_squared())
        self.adjoint_problem.has_solution = True

        self.print_results()

        while True:

            for j in range(len(self.controls)):
                self.controls[j].vector()[
                    self.idx_active_lower[j]
                ] = self.control_constraints[j][0].vector()[self.idx_active_lower[j]]
                self.controls[j].vector()[
                    self.idx_active_upper[j]
                ] = self.control_constraints[j][1].vector()[self.idx_active_upper[j]]

            self.inner_solver.run(self.idx_active)

            for j in range(len(self.controls)):
                self.mu[j].vector().vec().aypx(0.0, -self.gradients[j].vector().vec())
                self.mu[j].vector()[self.idx_inactive[j]] = 0.0

            self.objective_value = self.inner_solver.line_search.objective_step
            norm = np.sqrt(self._stationary_measure_squared())

            self.relative_norm = norm / norm_init

            self.compute_active_inactive_sets()

            self.iteration += 1

            if self.converged:
                break

            if self.iteration >= self.maximum_iterations:
                self.converged_reason = -1
                break

            self.print_results()
