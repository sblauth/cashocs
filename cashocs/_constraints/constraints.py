"""
Created on 05/11/2021, 09.40

@author: blauths
"""

import abc

import fenics
import numpy as np

from .._exceptions import InputError
from ..utils import _max, _min


class Constraint(abc.ABC):
    def __init__(self, variable_function, measure=None):
        self.variable_function = variable_function
        self.measure = measure

        self.is_integral_constraint = False
        self.is_pointwise_constraint = False

        if self.measure is None:
            self.is_integral_constraint = True
        else:
            self.is_pointwise_constraint = True

        self.linear_term = None
        self.quadratic_term = None

    @abc.abstractmethod
    def constraint_violation(self):
        pass


class EqualityConstraint(Constraint):
    def __init__(self, variable_function, target, measure=None):
        super().__init__(variable_function, measure=measure)
        self.target = target

        if self.is_integral_constraint:
            self.linear_term = variable_function
            self.quadratic_term = {
                "integrand": variable_function,
                "tracking_goal": target,
                "weight": 1.0,
            }

        elif self.is_pointwise_constraint:
            mesh = measure.ufl_domain().ufl_cargo()
            multiplier_space = fenics.FunctionSpace(mesh, "CG", 1)
            self.multiplier = fenics.Function(multiplier_space)
            self.linear_term = self.multiplier * variable_function * measure
            self.quadratic_term = (
                fenics.Constant(0.5) * pow(variable_function - target, 2) * measure
            )

    def constraint_violation(self):
        if self.is_integral_constraint:
            return np.abs(fenics.assemble(self.variable_function) - self.target)
        elif self.is_pointwise_constraint:
            return np.sqrt(
                fenics.assemble(
                    pow(self.variable_function - self.target, 2) * self.measure
                )
            )


class InequalityConstraint(Constraint):
    def __init__(
        self, variable_function, lower_bound=None, upper_bound=None, measure=None
    ):
        super().__init__(variable_function, measure=measure)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if self.lower_bound is None and self.upper_bound is None:
            raise InputError(
                "cashocs._constraints.constraints.InequalityConstraint",
                "lower_bound and upper_bound",
                "You have to specify at least one bound for the inequality constraint.",
            )

        if self.is_integral_constraint:
            self.min_max_term = {
                "integrand": variable_function,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "mu": 1.0,
                "lambda": 1.0,
            }

        elif self.is_pointwise_constraint:
            mesh = measure.ufl_domain().ufl_cargo()
            multiplier_space = fenics.FunctionSpace(mesh, "CG", 1)
            self.multiplier = fenics.Function(multiplier_space)
            weight_space = fenics.FunctionSpace(mesh, "R", 0)
            self.weight = fenics.Function(weight_space)

            self.cost_functional_terms = []
            if self.upper_bound is not None:
                self.cost_functional_terms.append(
                    fenics.Constant(1 / 2)
                    / self.weight
                    * pow(
                        _max(
                            fenics.Constant(0.0),
                            self.multiplier
                            + self.weight * (self.variable_function - self.upper_bound),
                        ),
                        2,
                    )
                    * self.measure
                )

            if self.lower_bound is not None:
                self.cost_functional_terms.append(
                    fenics.Constant(1 / 2)
                    / self.weight
                    * pow(
                        _min(
                            fenics.Constant(0.0),
                            self.multiplier
                            + self.weight * (self.variable_function - self.lower_bound),
                        ),
                        2,
                    )
                    * self.measure
                )

    def constraint_violation(self):
        violation = 0.0
        if self.is_integral_constraint:
            min_max_integral = fenics.assemble(self.min_max_term["integrand"])

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
                        _max(
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
                        _min(
                            self.variable_function - self.lower_bound,
                            fenics.Constant(0.0),
                        ),
                        2,
                    )
                    * self.measure
                )

        return np.sqrt(violation)
