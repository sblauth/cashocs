"""
Created on 05/11/2021, 09.36

@author: blauths
"""


class ConstrainedOptimizationAlgorithm:
    def __init__(self, optimization_problem, constraints):
        self.optimization_problem = optimization_problem
        self.constraints = constraints


class AugmentedLagrangianMethod(ConstrainedOptimizationAlgorithm):
    def __init__(self, optimization_problem, constraints, gamma_0=None, lambda_0=None):
        """

        Parameters
        ----------
        optimization_problem : cashocs.OptimalControlProblem or cashocs.ShapeOptimizationProblem
        constraints
        gamma_0
        lambda_0
        """
        super().__init__(optimization_problem, constraints)

        if not isinstance(gamma_0, list):
            self.gamma_0 = [gamma_0]
        else:
            self.gamma_0 = gamma_0

        if not isinstance(lambda_0, list):
            self.lambda_0 = [lambda_0]
        else:
            self.lambda_0 = lambda_0

        self.cost_functional = optimization_problem.reduced_cost_functional


class LagrangianMethod(ConstrainedOptimizationAlgorithm):
    def __init__(self, optimization_problem, constraints):
        super().__init__(optimization_problem, constraints)
        pass


class QuadraticPenaltyMethod(ConstrainedOptimizationAlgorithm):
    def __init__(self, optimization_problem, constraints):
        super().__init__(optimization_problem, constraints)
        pass


class L1PenaltyMethod(ConstrainedOptimizationAlgorithm):
    def __init__(self, optimization_problem, constraints):
        super().__init__(optimization_problem, constraints)
