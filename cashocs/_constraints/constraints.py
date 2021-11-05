"""
Created on 05/11/2021, 09.40

@author: blauths
"""

import numpy as np
import fenics


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
            self.linear_term = variable_function * measure
            self.quadratic_term = (
                fenics.Constant(0.5) * pow(variable_function - target, 2) * measure
            )

    def constraint_violation(self):
        if self.is_integral_constraint:
            return np.abs(fenics.assemble(self.variable_function) - self.target)
        elif self.is_pointwise_constraint:
            return np.sqrt(
                fenics.assemble(pow(self.variable_function - self.target, 2))
            )


class InequalityConstraint(Constraint):
    def __init__(
        self, variable_function, lower_bound=None, upper_bound=None, measure=None
    ):
        super().__init__(variable_function, measure=measure)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
