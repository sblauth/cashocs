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

"""Equality and inequality constraints."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    try:
        from ufl_legacy.core import expr as ufl_expr
    except ImportError:
        from ufl.core import expr as ufl_expr


class Constraint(abc.ABC):
    """Base class for additional equality and inequality constraints."""

    multiplier: fenics.Function
    target: float
    min_max_term: cost_functional.MinMaxFunctional
    lower_bound: fenics.Function | float
    upper_bound: fenics.Function | float

    def __init__(
        self,
        variable_function: ufl.Form | ufl_expr.Expr,
        measure: ufl.Measure | None = None,
    ) -> None:
        """Initializes self.

        Args:
            variable_function: Either a UFL Form (when we have a scalar / integral
                constraint) or an ufl expression (when we have a pointwise constraint),
                which models the part that is to be constrained.
            measure: A measure indicating where a pointwise constraint should be
                satisfied.

        """
        self.variable_function = variable_function
        self.measure = measure

        self.is_integral_constraint = False
        self.is_pointwise_constraint = False

        if self.measure is None:
            self.is_integral_constraint = True
        else:
            self.is_pointwise_constraint = True

    @abc.abstractmethod
    def constraint_violation(self) -> float:
        """Computes the constraint violation for the problem.

        Returns:
            The computed violation

        """
        pass


class EqualityConstraint(Constraint):
    """Models an (additional) equality constraint."""

    def __init__(
        self,
        variable_function: ufl.Form | ufl_expr.Expr,
        target: float,
        measure: ufl.Measure | None = None,
    ) -> None:
        """Initializes self.

        Args:
            variable_function: Either a UFL Form (when we have a scalar / integral
                constraint) or an ufl expression (when we have a pointwise constraint),
                which models the part that is to be constrained.
            target: The target (rhs) of the equality constraint.
            measure: A measure indicating where a pointwise constraint should be
                satisfied.

        """
        super().__init__(variable_function, measure=measure)
        self.target = target

        if self.is_integral_constraint:
            self.linear_form = variable_function
            self.quadratic_functional = cost_functional.ScalarTrackingFunctional(
                variable_function, target, 1.0
            )

        elif self.measure is not None:
            mesh = self.measure.ufl_domain().ufl_cargo()
            multiplier_space = fenics.FunctionSpace(mesh, "CG", 1)
            self.multiplier = fenics.Function(multiplier_space)
            self.linear_functional = cost_functional.IntegralFunctional(
                self.multiplier * variable_function * measure
            )
            self.quadratic_form = (
                fenics.Constant(0.5) * pow(variable_function - target, 2) * measure
            )

    def constraint_violation(self) -> float:
        """Computes the constraint violation for the problem.

        Returns:
            The computed violation

        """
        violation = float("inf")
        if self.is_integral_constraint:
            violation = np.abs(fenics.assemble(self.variable_function) - self.target)
        elif self.is_pointwise_constraint:
            violation = np.sqrt(
                fenics.assemble(
                    pow(self.variable_function - self.target, 2) * self.measure
                )
            )
        return violation


class InequalityConstraint(Constraint):
    """Models an (additional) inequality constraint."""

    def __init__(
        self,
        variable_function: ufl.Form | ufl_expr.Expr,
        lower_bound: float | fenics.Function | None = None,
        upper_bound: float | fenics.Function | None = None,
        measure: ufl.Measure | None = None,
    ) -> None:
        """Initializes self.

        Args:
            variable_function: Either a UFL Form (when we have a scalar / integral
                constraint) or an ufl expression (when we have a pointwise constraint),
                which models the part that is to be constrained
            lower_bound: The lower bound for the inequality constraint
            upper_bound: The upper bound for the inequality constraint
            measure: A measure indicating where a pointwise constraint should be
                satisfied.

        """
        super().__init__(variable_function, measure=measure)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if self.lower_bound is None and self.upper_bound is None:
            raise _exceptions.InputError(
                "cashocs._constraints.constraints.InequalityConstraint",
                "lower_bound and upper_bound",
                "You have to specify at least one bound for the inequality constraint.",
            )

        if self.is_integral_constraint:
            self.min_max_term = cost_functional.MinMaxFunctional(
                variable_function, lower_bound, upper_bound, 1.0, 1.0
            )

        elif self.measure is not None:
            mesh = self.measure.ufl_domain().ufl_cargo()
            multiplier_space = fenics.FunctionSpace(mesh, "CG", 1)
            self.multiplier = fenics.Function(multiplier_space)
            self.weight = fenics.Constant(0.0)

            self.cost_functional_terms = []
            if self.upper_bound is not None:
                upper_functional = cost_functional.IntegralFunctional(
                    fenics.Constant(1 / 2)
                    / self.weight
                    * pow(
                        _utils.max_(
                            fenics.Constant(0.0),
                            self.multiplier
                            + self.weight * (self.variable_function - self.upper_bound),
                        ),
                        2,
                    )
                    * self.measure
                )
                self.cost_functional_terms.append(upper_functional)

            if self.lower_bound is not None:
                lower_functional = cost_functional.IntegralFunctional(
                    fenics.Constant(1 / 2)
                    / self.weight
                    * pow(
                        _utils.min_(
                            fenics.Constant(0.0),
                            self.multiplier
                            + self.weight * (self.variable_function - self.lower_bound),
                        ),
                        2,
                    )
                    * self.measure
                )
                self.cost_functional_terms.append(lower_functional)

    def constraint_violation(self) -> float:
        """Computes the constraint violation for the problem.

        Returns:
            The computed violation

        """
        violation: float = 0.0
        if self.is_integral_constraint:
            min_max_integral = fenics.assemble(self.min_max_term.integrand)

            if self.upper_bound is not None:
                violation += pow(
                    np.maximum(min_max_integral - self.upper_bound, 0.0), 2
                )

            if self.lower_bound is not None:
                violation += pow(
                    np.minimum(min_max_integral - self.lower_bound, 0.0), 2
                )

        elif self.is_pointwise_constraint:
            if self.upper_bound is not None:
                violation += fenics.assemble(
                    pow(
                        _utils.max_(
                            self.variable_function - self.upper_bound,
                            fenics.Constant(0.0),
                        ),
                        2,
                    )
                    * self.measure
                )

            if self.lower_bound is not None:
                violation += fenics.assemble(
                    pow(
                        _utils.min_(
                            self.variable_function - self.lower_bound,
                            fenics.Constant(0.0),
                        ),
                        2,
                    )
                    * self.measure
                )

        violation = np.sqrt(violation)

        return violation
