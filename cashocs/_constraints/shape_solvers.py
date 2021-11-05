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

from .._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .solvers import AugmentedLagrangianProblem
from .._loggers import debug


class AugmentedLagrangianShapeOptimizationProblem(AugmentedLagrangianProblem):
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        adjoints,
        boundaries,
        constraints,
        config=None,
        shape_scalar_product=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        desired_weights=None,
        scalar_tracking_forms=None,
        mu_0=None,
        lambda_0=None,
    ):
        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            constraints,
            config=config,
            initial_guess=initial_guess,
            ksp_options=ksp_options,
            adjoint_ksp_options=adjoint_ksp_options,
            desired_weights=desired_weights,
            scalar_tracking_forms=scalar_tracking_forms,
            mu_0=mu_0,
            lambda_0=lambda_0,
        )

        self.boundaries = boundaries
        self.shape_scalar_product = shape_scalar_product

    def solve(self, tol=1e-2, max_iter=10):
        super().solve(tol=tol, max_iter=max_iter)

    def _solve_inner_problem(self, tol=1e-2):
        sop = ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.adjoints,
            self.boundaries,
            config=self.config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
            scalar_tracking_forms=self.scalar_tracking_forms,
        )
        sop.solve(rtol=tol, atol=tol / 10.0)
