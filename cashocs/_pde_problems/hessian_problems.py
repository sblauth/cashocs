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

"""Hessian problem.

This uses Krylov subspace methods to iteratively solve the "Hessian problems" occurring
in the truncated Newton method.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING, Union

import fenics
import numpy as np

from cashocs import _loggers
from cashocs import _utils
from cashocs import nonlinear_solvers

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization.optimal_control import box_constraints as boxc
    from cashocs._pde_problems import control_gradient_problem


class HessianProblem:
    """PDE Problem used to solve the (reduced) Hessian problem."""

    def __init__(
        self,
        db: database.Database,
        form_handler: _forms.ControlFormHandler,
        adjoint_form_handler: _forms.AdjointFormHandler,
        gradient_problem: control_gradient_problem.ControlGradientProblem,
        box_constraints: boxc.BoxConstraints,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The FormHandler object for the optimization problem.
            adjoint_form_handler: The form handler for the adjoint system.
            gradient_problem: The ControlGradientProblem object (this is needed for the
                computation of the Hessian).
            box_constraints: The box constraints for the problem.

        """
        self.db = db
        self.form_handler = form_handler
        self.adjoint_form_handler = adjoint_form_handler
        self.gradient_problem = gradient_problem
        self.box_constraints = box_constraints

        self.config = self.db.config

        self.inner_newton = self.config.get("AlgoTNM", "inner_newton")
        self.max_it_inner_newton = self.config.getint("AlgoTNM", "max_it_inner_newton")
        self.inner_newton_rtol = self.config.getfloat("AlgoTNM", "inner_newton_rtol")
        self.inner_newton_atol = self.config.getfloat("AlgoTNM", "inner_newton_atol")

        self.test_directions = self.form_handler.hessian_form_handler.test_directions

        self.residual = _utils.create_function_list(self.db.function_db.control_spaces)
        self.delta_control = _utils.create_function_list(
            self.db.function_db.control_spaces
        )
        self.temp1 = _utils.create_function_list(self.db.function_db.control_spaces)
        self.temp2 = _utils.create_function_list(self.db.function_db.control_spaces)
        self.p = _utils.create_function_list(self.db.function_db.control_spaces)
        self.p_prev = _utils.create_function_list(self.db.function_db.control_spaces)
        self.p_pprev = _utils.create_function_list(self.db.function_db.control_spaces)
        self.s = _utils.create_function_list(self.db.function_db.control_spaces)
        self.s_prev = _utils.create_function_list(self.db.function_db.control_spaces)
        self.s_pprev = _utils.create_function_list(self.db.function_db.control_spaces)
        self.q = _utils.create_function_list(self.db.function_db.control_spaces)
        self.q_prev = _utils.create_function_list(self.db.function_db.control_spaces)
        self.hessian_actions = _utils.create_function_list(
            self.db.function_db.control_spaces
        )
        self.inactive_part = _utils.create_function_list(
            self.db.function_db.control_spaces
        )
        self.active_part = _utils.create_function_list(
            self.db.function_db.control_spaces
        )

        # pylint: disable=invalid-name
        self.state_A_tensors = [
            fenics.PETScMatrix() for _ in range(self.db.parameter_db.state_dim)
        ]
        self.state_b_tensors = [
            fenics.PETScVector() for _ in range(self.db.parameter_db.state_dim)
        ]
        # pylint: disable=invalid-name
        self.adjoint_A_tensors = [
            fenics.PETScMatrix() for _ in range(self.db.parameter_db.state_dim)
        ]
        self.adjoint_b_tensors = [
            fenics.PETScVector() for _ in range(self.db.parameter_db.state_dim)
        ]

        self.state_dim = self.db.parameter_db.state_dim

        self.picard_rtol = self.config.getfloat("StateSystem", "picard_rtol")
        self.picard_atol = self.config.getfloat("StateSystem", "picard_atol")
        self.picard_max_iter = self.config.getint("StateSystem", "picard_iter")
        self.picard_verbose = self.config.getboolean("StateSystem", "picard_verbose")

        self.no_sensitivity_solves = 0

        self.bcs_list_ad = self.adjoint_form_handler.bcs_list_ad

        option: List[List[Union[str, int, float]]] = [
            ["ksp_type", "cg"],
            ["pc_type", "hypre"],
            ["pc_hypre_type", "boomeramg"],
            ["pc_hypre_boomeramg_strong_threshold", 0.7],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-50],
            ["ksp_max_it", 100],
        ]
        self.riesz_ksp_options: _typing.KspOptions = []
        for _ in range(len(self.db.function_db.controls)):
            self.riesz_ksp_options.append(option)

    def hessian_application(
        self, h: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        r"""Computes the application of the Hessian to some element.

        This is needed in the truncated Newton method where we solve the system

        .. math:: J''(u) [\Delta u] = - J'(u)

        via iterative methods (conjugate gradient or conjugate residual method)

        Args:
            h: A function to which we want to apply the Hessian to.
            out: A list of functions into which the result is saved.

        """
        for i in range(len(self.test_directions)):
            self.test_directions[i].vector().vec().aypx(0.0, h[i].vector().vec())
            self.test_directions[i].vector().apply("")

        self.bcs_list_ad = self.adjoint_form_handler.bcs_list_ad

        if (
            not self.config.getboolean("StateSystem", "picard_iteration")
            or self.state_dim == 1
        ):

            for i in range(self.state_dim):
                _utils.assemble_and_solve_linear(
                    self.form_handler.hessian_form_handler.sensitivity_eqs_lhs[i],
                    self.form_handler.hessian_form_handler.sensitivity_eqs_rhs[i],
                    self.bcs_list_ad[i],
                    x=self.db.function_db.states_prime[i].vector().vec(),
                    ksp_options=self.db.parameter_db.state_ksp_options[i],
                )
                self.db.function_db.states_prime[i].vector().apply("")

            for i in range(self.state_dim):
                _utils.assemble_and_solve_linear(
                    self.form_handler.hessian_form_handler.adjoint_sensitivity_eqs_lhs[
                        -1 - i
                    ],
                    self.form_handler.hessian_form_handler.w_1[-1 - i],
                    self.bcs_list_ad[-1 - i],
                    x=self.db.function_db.adjoints_prime[-1 - i].vector().vec(),
                    ksp_options=self.db.parameter_db.adjoint_ksp_options[-1 - i],
                )
                self.db.function_db.adjoints_prime[-1 - i].vector().apply("")

        else:
            nonlinear_solvers.picard_iteration(
                self.form_handler.hessian_form_handler.sensitivity_eqs_picard,
                self.db.function_db.states_prime,
                self.adjoint_form_handler.bcs_list_ad,
                max_iter=self.picard_max_iter,
                rtol=self.picard_rtol,
                atol=self.picard_atol,
                verbose=self.picard_verbose,
                inner_damped=False,
                inner_inexact=False,
                inner_verbose=False,
                inner_max_its=2,
                ksp_options=self.db.parameter_db.state_ksp_options,
                A_tensors=self.state_A_tensors,
                b_tensors=self.state_b_tensors,
                inner_is_linear=True,
            )

            nonlinear_solvers.picard_iteration(
                self.form_handler.hessian_form_handler.adjoint_sensitivity_eqs_picard,
                self.db.function_db.adjoints_prime,
                self.adjoint_form_handler.bcs_list_ad,
                max_iter=self.picard_max_iter,
                rtol=self.picard_rtol,
                atol=self.picard_atol,
                verbose=self.picard_verbose,
                inner_damped=False,
                inner_inexact=False,
                inner_verbose=False,
                inner_max_its=2,
                ksp_options=self.db.parameter_db.adjoint_ksp_options,
                A_tensors=self.adjoint_A_tensors,
                b_tensors=self.adjoint_b_tensors,
                inner_is_linear=True,
            )

        for i in range(len(out)):
            b = fenics.as_backend_type(
                fenics.assemble(self.form_handler.hessian_form_handler.hessian_rhs[i])
            ).vec()

            _utils.solve_linear_problem(
                A=self.form_handler.riesz_projection_matrices[i],
                b=b,
                x=out[i].vector().vec(),
                ksp_options=self.riesz_ksp_options[i],
            )
            out[i].vector().apply("")

        self.no_sensitivity_solves += 2

    def reduced_hessian_application(
        self, h: List[fenics.Function], out: List[fenics.Function]
    ) -> None:
        """Computes the application of the reduced Hessian on a direction.

        This is needed to solve the Newton step with iterative solvers.

        Args:
            h: The direction, onto which the reduced Hessian is applied.
            out: The output of the application of the (linear) operator.

        """
        for j in range(len(out)):
            out[j].vector().vec().set(0.0)
            out[j].vector().apply("")

        self.box_constraints.restrictor.restrict_to_inactive_set(h, self.inactive_part)
        self.hessian_application(self.inactive_part, self.hessian_actions)
        self.box_constraints.restrictor.restrict_to_inactive_set(
            self.hessian_actions, self.inactive_part
        )
        self.box_constraints.restrictor.restrict_to_active_set(h, self.active_part)

        for j in range(len(out)):
            out[j].vector().vec().aypx(
                0.0,
                self.active_part[j].vector().vec()
                + self.inactive_part[j].vector().vec(),
            )
            out[j].vector().apply("")

    def newton_solve(self) -> List[fenics.Function]:
        """Solves the Newton step with an iterative method.

        Returns:
            A list containing the Newton increment.

        """
        self.gradient_problem.solve()
        self.box_constraints.restrictor.compute_active_sets()

        for fun in self.delta_control:
            fun.vector().vec().set(0.0)
            fun.vector().apply("")

        if self.inner_newton.casefold() == "cg":
            self.cg()
        elif self.inner_newton.casefold() == "cr":
            self.cr()

        return self.delta_control

    def cg(self) -> None:
        """Solves the (truncated) Newton step with a CG method."""
        for j in range(len(self.residual)):
            self.residual[j].vector().vec().aypx(
                0.0, -self.db.function_db.gradient[j].vector().vec()
            )
            self.residual[j].vector().apply("")
            self.p[j].vector().vec().aypx(0.0, self.residual[j].vector().vec())
            self.p[j].vector().apply("")

        rsold = self.form_handler.scalar_product(self.residual, self.residual)
        eps_0 = np.sqrt(rsold)

        for _ in range(self.max_it_inner_newton):

            self.reduced_hessian_application(self.p, self.q)

            self.box_constraints.restrictor.restrict_to_active_set(self.p, self.temp1)
            sp_val1 = self.form_handler.scalar_product(self.temp1, self.temp1)

            self.box_constraints.restrictor.restrict_to_inactive_set(self.p, self.temp1)
            self.box_constraints.restrictor.restrict_to_inactive_set(self.q, self.temp2)
            sp_val2 = self.form_handler.scalar_product(self.temp1, self.temp2)
            sp_val = sp_val1 + sp_val2
            alpha = rsold / sp_val

            for j in range(len(self.delta_control)):
                self.delta_control[j].vector().vec().axpy(
                    alpha, self.p[j].vector().vec()
                )
                self.delta_control[j].vector().apply("")
                self.residual[j].vector().vec().axpy(-alpha, self.q[j].vector().vec())
                self.residual[j].vector().apply("")

            rsnew = self.form_handler.scalar_product(self.residual, self.residual)
            eps = np.sqrt(rsnew)
            _loggers.debug(f"Residual of the CG method: {eps/eps_0:.3e} (relative)")
            if eps < self.inner_newton_atol + self.inner_newton_rtol * eps_0:
                break

            beta = rsnew / rsold

            for j in range(len(self.p)):
                self.p[j].vector().vec().aypx(beta, self.residual[j].vector().vec())
                self.p[j].vector().apply("")

            rsold = rsnew

    def cr(self) -> None:
        """Solves the (truncated) Newton step with a CR method."""
        for j in range(len(self.residual)):
            self.residual[j].vector().vec().aypx(
                0.0, -self.db.function_db.gradient[j].vector().vec()
            )
            self.residual[j].vector().apply("")
            self.p[j].vector().vec().aypx(0.0, self.residual[j].vector().vec())
            self.p[j].vector().apply("")

        eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

        self.reduced_hessian_application(self.residual, self.s)

        for j in range(len(self.q)):
            self.q[j].vector().vec().aypx(0.0, self.s[j].vector().vec())
            self.q[j].vector().apply("")

        self.box_constraints.restrictor.restrict_to_active_set(
            self.residual, self.temp1
        )
        self.box_constraints.restrictor.restrict_to_active_set(self.s, self.temp2)
        sp_val1 = self.form_handler.scalar_product(self.temp1, self.temp2)
        self.box_constraints.restrictor.restrict_to_inactive_set(
            self.residual, self.temp1
        )
        self.box_constraints.restrictor.restrict_to_inactive_set(self.s, self.temp2)
        sp_val2 = self.form_handler.scalar_product(self.temp1, self.temp2)

        rar = sp_val1 + sp_val2

        for i in range(self.max_it_inner_newton):

            self.box_constraints.restrictor.restrict_to_active_set(self.q, self.temp1)
            self.box_constraints.restrictor.restrict_to_inactive_set(self.q, self.temp2)
            denom1 = self.form_handler.scalar_product(self.temp1, self.temp1)
            denom2 = self.form_handler.scalar_product(self.temp2, self.temp2)
            denominator = denom1 + denom2

            alpha = rar / denominator

            for j in range(len(self.delta_control)):
                self.delta_control[j].vector().vec().axpy(
                    alpha, self.p[j].vector().vec()
                )
                self.delta_control[j].vector().apply("")
                self.residual[j].vector().vec().axpy(-alpha, self.q[j].vector().vec())
                self.residual[j].vector().apply("")

            eps = np.sqrt(
                self.form_handler.scalar_product(self.residual, self.residual)
            )
            _loggers.debug(f"Residual of the CR method: {eps/eps_0:.3e} (relative)")
            if (
                eps < self.inner_newton_atol + self.inner_newton_rtol * eps_0
                or i == self.max_it_inner_newton - 1
            ):
                break

            self.reduced_hessian_application(self.residual, self.s)

            self.box_constraints.restrictor.restrict_to_active_set(
                self.residual, self.temp1
            )
            self.box_constraints.restrictor.restrict_to_active_set(self.s, self.temp2)
            sp_val1 = self.form_handler.scalar_product(self.temp1, self.temp2)
            self.box_constraints.restrictor.restrict_to_inactive_set(
                self.residual, self.temp1
            )
            self.box_constraints.restrictor.restrict_to_inactive_set(self.s, self.temp2)
            sp_val2 = self.form_handler.scalar_product(self.temp1, self.temp2)

            rar_new = sp_val1 + sp_val2
            beta = rar_new / rar

            for j in range(len(self.p)):
                self.p[j].vector().vec().aypx(beta, self.residual[j].vector().vec())
                self.p[j].vector().apply("")
                self.q[j].vector().vec().aypx(beta, self.s[j].vector().vec())
                self.q[j].vector().apply("")

            rar = rar_new
