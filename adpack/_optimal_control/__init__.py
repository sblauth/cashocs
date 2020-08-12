"""Treatment of optimal control problems.

This module is used for the treatment of optimal control problems.
It includes the optimization problem, the solution algorithms and
the line search needed for this.
"""

from .cost_functional import ReducedCostFunctional
from .line_search import ArmijoLineSearch
from .optimization_algorithm import OptimizationAlgorithm

__all__ = ['ReducedCostFunctional', 'ArmijoLineSearch', 'OptimizationAlgorithm']
