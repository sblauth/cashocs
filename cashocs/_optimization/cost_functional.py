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

"""Reduced cost functionals."""

from __future__ import annotations

import abc
import ctypes
from typing import cast, TYPE_CHECKING

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _forms
from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import _pde_problems
    from cashocs import _typing
    from cashocs._database import database


class ReducedCostFunctional:
    """Reduced cost functional for PDE constrained optimization."""

    def __init__(
        self,
        db: database.Database,
        form_handler: _typing.FormHandler,
        state_problem: _pde_problems.StateProblem,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The FormHandler object for the optimization problem.
            state_problem: The StateProblem object corresponding to the state system.

        """
        self.db = db
        self.form_handler = form_handler
        self.state_problem = state_problem

    def evaluate(self) -> float:
        """Evaluates the reduced cost functional.

        First solves the state system, so that the state variables are up-to-date,
        and then evaluates the reduced cost functional by assembling the corresponding
        UFL form.

        Returns:
            The value of the reduced cost functional

        """
        self.state_problem.solve()

        vals = [
            functional.evaluate() for functional in self.db.form_db.cost_functional_list
        ]
        val = sum(vals)
        val += self.form_handler.cost_functional_shift

        if self.db.parameter_db.problem_type == "shape":
            self.form_handler = cast(_forms.ShapeFormHandler, self.form_handler)
            val += self.form_handler.shape_regularization.compute_objective()

        return val


class Functional(abc.ABC):
    """Base class for all cost functionals."""

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Evaluates the functional.

        Returns:
            The current value of the functional.

        """
        pass

    @abc.abstractmethod
    def derivative(
        self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr
    ) -> ufl.Form:
        """Computes the derivative of the functional w.r.t. argument towards direction.

        Args:
            argument: The argument, w.r.t. which the functional is differentiated
            direction: The direction into which the derivative is computed

        Returns:
            A form of the resulting derivative

        """
        pass

    @abc.abstractmethod
    def coefficients(self) -> tuple[fenics.Function]:
        """Computes the ufl coefficients which are used in the functional.

        Returns:
            The set of used coefficients.

        """
        pass

    @abc.abstractmethod
    def scale(self, scaling_factor: float | int) -> None:
        """Scales the functional by a scalar.

        Args:
            scaling_factor: The scaling factor used to scale the functional

        """
        pass

    @abc.abstractmethod
    def update(self) -> None:
        """Updates the functional after solving the state equation."""
        pass


class IntegralFunctional(Functional):
    """A functional which is given by the integral of ``form``."""

    def __init__(self, form: ufl.Form) -> None:
        """Initializes self.

        Args:
            form: The form of the integrand, which is to be calculated for evaluating
                the functional.

        """
        super().__init__()
        self.form = form

    def evaluate(self) -> float:
        """Evaluates the functional.

        Returns:
            The current value of the functional.

        """
        val: float = fenics.assemble(self.form)
        return val

    def derivative(
        self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr
    ) -> ufl.Form:
        """Computes the derivative of the functional w.r.t. argument towards direction.

        Args:
            argument: The argument, w.r.t. which the functional is differentiated
            direction: The direction into which the derivative is computed

        Returns:
            A form of the resulting derivative

        """
        return fenics.derivative(self.form, argument, direction)

    def coefficients(self) -> tuple[fenics.Function]:
        """Computes the ufl coefficients which are used in the functional.

        Returns:
            The set of used coefficients.

        """
        coeffs: tuple[fenics.Function] = self.form.coefficients()
        return coeffs

    def scale(self, scaling_factor: float | int) -> None:
        """Scales the functional by a scalar.

        Args:
            scaling_factor: The scaling factor used to scale the functional

        """
        self.form = fenics.Constant(scaling_factor) * self.form

    def update(self) -> None:
        """Updates the functional after solving the state equation."""
        pass


class ScalarTrackingFunctional(Functional):
    """Tracking cost functional for scalar quantities arising due to integration."""

    def __init__(
        self,
        integrand: ufl.Form,
        tracking_goal: float | int | ctypes.c_float | ctypes.c_double,
        weight: float | int = 1.0,
    ) -> None:
        """Initializes self.

        Args:
            integrand: The integrand of the functional
            tracking_goal: A real number, which the integral of the integrand should
                track. Note, that when a ctypes object is passed, the float is assumed
                to be mutable and the tracking_goal is updated every iteration.
            weight: A real number which gives the scaling factor for this functional

        """
        super().__init__()
        self.integrand = integrand
        self.tracking_goal = tracking_goal
        if not isinstance(self.tracking_goal, (ctypes.c_float, ctypes.c_double)):
            self.tracking_goal_value = self.tracking_goal
        else:
            self.tracking_goal_value = self.tracking_goal.value

        self.integrand_value = fenics.Constant(0.0)
        self.weight = fenics.Constant(weight)

    def evaluate(self) -> float:
        """Evaluates the functional.

        Returns:
            The current value of the functional.

        """
        if isinstance(self.tracking_goal, (ctypes.c_float, ctypes.c_double)):
            self.tracking_goal_value = self.tracking_goal.value

        scalar_integral_value = fenics.assemble(self.integrand)
        self.integrand_value.assign(scalar_integral_value)
        val: float = (
            self.weight.values()[0]
            / 2.0
            * pow(scalar_integral_value - self.tracking_goal_value, 2)
        )
        return val

    def derivative(
        self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr
    ) -> ufl.Form:
        """Computes the derivative of the functional w.r.t. argument towards direction.

        Args:
            argument: The argument, w.r.t. which the functional is differentiated
            direction: The direction into which the derivative is computed

        Returns:
            A form of the resulting derivative

        """
        if isinstance(self.tracking_goal, (ctypes.c_float, ctypes.c_double)):
            self.tracking_goal_value = self.tracking_goal.value

        derivative = fenics.derivative(
            self.weight
            * (self.integrand_value - fenics.Constant(self.tracking_goal_value))
            * self.integrand,
            argument,
            direction,
        )
        return derivative

    def coefficients(self) -> tuple[fenics.Function]:
        """Computes the ufl coefficients which are used in the functional.

        Returns:
            The set of used coefficients.

        """
        coeffs: tuple[fenics.Function] = self.integrand.coefficients()
        return coeffs

    def scale(self, scaling_factor: float | int) -> None:
        """Scales the functional by a scalar.

        Args:
            scaling_factor: The scaling factor used to scale the functional

        """
        self.weight.assign(scaling_factor)

    def update(self) -> None:
        """Updates the functional after solving the state equation."""
        scalar_integral_value = fenics.assemble(self.integrand)
        self.integrand_value.assign(scalar_integral_value)


class MinMaxFunctional(Functional):
    """Cost functional involving a maximum of 0 and a integral term squared."""

    def __init__(
        self,
        integrand: ufl.Form,
        lower_bound: float | int | None = None,
        upper_bound: float | int | None = None,
        mu: float | int = 1.0,
        lambd: float | int = 0.0,
    ) -> None:
        """Initializes self."""
        super().__init__()
        self.integrand = integrand

        if upper_bound is None and lower_bound is None:
            raise _exceptions.InputError(
                "cashocs.MinMaxFunctional",
                "lower_bound and upper_bound",
                "At least one of lower_bound or upper_bound must not be None.",
            )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.integrand_value = fenics.Constant(0.0)
        self.mu = fenics.Constant(mu)
        self.lambd = fenics.Constant(lambd)

    def evaluate(self) -> float:
        """Evaluates the functional.

        Returns:
            The current value of the functional.

        """
        min_max_integrand_value = fenics.assemble(self.integrand)
        self.integrand_value.assign(min_max_integrand_value)

        val = 0.0
        if self.lower_bound is not None:
            val += (
                1
                / (2 * self.mu.values()[0])
                * pow(
                    np.minimum(
                        0.0,
                        self.lambd.values()[0]
                        + self.mu.values()[0]
                        * (min_max_integrand_value - self.lower_bound),
                    ),
                    2,
                )
            )
        if self.upper_bound is not None:
            val += (
                1
                / (2 * self.mu.values()[0])
                * pow(
                    np.maximum(
                        0.0,
                        self.lambd.values()[0]
                        + self.mu.values()[0]
                        * (min_max_integrand_value - self.upper_bound),
                    ),
                    2,
                )
            )
        return val

    def derivative(
        self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr
    ) -> ufl.Form:
        """Computes the derivative of the functional w.r.t. argument towards direction.

        Args:
            argument: The argument, w.r.t. which the functional is differentiated
            direction: The direction into which the derivative is computed

        Returns:
            A form of the resulting derivative

        """
        derivative_list = []
        if self.lower_bound is not None:
            term_lower = self.lambd + self.mu * (
                self.integrand_value - self.lower_bound
            )
            derivative = fenics.derivative(
                _utils.min_(fenics.Constant(0.0), term_lower) * self.integrand,
                argument,
                direction,
            )
            derivative_list.append(derivative)

        if self.upper_bound is not None:
            term_upper = self.lambd + self.mu * (
                self.integrand_value - self.upper_bound
            )
            derivative = fenics.derivative(
                _utils.max_(fenics.Constant(0.0), term_upper) * self.integrand,
                argument,
                direction,
            )
            derivative_list.append(derivative)

        return _utils.summation(derivative_list)

    def coefficients(self) -> tuple[fenics.Function]:
        """Computes the ufl coefficients which are used in the functional.

        Returns:
            The set of used coefficients.

        """
        coeffs: tuple[fenics.Function] = self.integrand.coefficients()
        return coeffs

    def scale(self, scaling_factor: float | int) -> None:
        """Scales the functional by a scalar.

        Args:
            scaling_factor: The scaling factor used to scale the functional

        """
        pass

    def update(self) -> None:
        """Updates the functional after solving the state equation."""
        min_max_integrand_value = fenics.assemble(self.integrand)
        self.integrand_value.assign(min_max_integrand_value)


class Lagrangian:
    """A Lagrangian function for the optimization problem."""

    def __init__(
        self,
        cost_functional_list: list[_typing.CostFunctional],
        state_forms: list[ufl.Form],
    ) -> None:
        """Initializes self.

        Args:
            cost_functional_list: The list of cost functionals.
            state_forms: The list of state forms.

        """
        self.cost_functional_list = cost_functional_list
        self.state_forms = state_forms
        self.summed_state_forms = _utils.summation(self.state_forms)

    def derivative(
        self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr
    ) -> ufl.Form:
        """Computes the derivative of the Lagrangian w.r.t. argument towards direction.

        Args:
            argument: The argument, w.r.t. which the Lagrangian is differentiated
            direction: The direction into which the derivative is computed

        Returns:
            A form of the resulting derivative

        """
        cost_functional_derivative_list = [
            functional.derivative(argument, direction)
            for functional in self.cost_functional_list
        ]
        cost_functional_derivative = _utils.summation(cost_functional_derivative_list)
        state_forms_derivative = fenics.derivative(
            self.summed_state_forms, argument, direction
        )
        derivative = cost_functional_derivative + state_forms_derivative
        return derivative

    def coefficients(self) -> set[fenics.Function]:
        """Computes the ufl coefficients which are used in the functional.

        Returns:
            The set of used coefficients.

        """
        state_coeffs = set(self.summed_state_forms.coefficients())
        functional_coeffs = [
            set(functional.coefficients()) for functional in self.cost_functional_list
        ]
        coeffs = set().union(*functional_coeffs)
        coeffs.union(state_coeffs)

        return coeffs
