"""
Created on 05/11/2021, 09.40

@author: blauths
"""

import numpy as np
import fenics

from .._exceptions import InputError


class Constraint:
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

        self.is_twosided = False
        self.is_onesided = False
        if self.lower_bound is not None and self.upper_bound is not None:
            self.is_twosided = True

        elif self.lower_bound is not None and self.upper_bound is None:
            self.is_onesided = True
            self.upper_bound = -self.lower_bound
            self.lower_bound = None

        elif self.upper_bound is not None and self.lower_bound is None:
            self.is_onesided = True

        else:
            raise InputError(
                "cashocs._constraints.constraints.InequalityConstraint",
                "lower_bound and upper_bound",
                "At least one of the bounds has to be given.",
            )

        if self.is_integral_constraint:
            if self.is_onesided:
                pass
