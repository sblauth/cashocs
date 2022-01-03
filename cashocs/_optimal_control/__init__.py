# Copyright (C) 2020-2022 Sebastian Blauth
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

"""Module for treatment of optimal control problems.

This module is used for the treatment of optimal control problems.
It includes the optimization problem, the solution algorithms and
the line search needed for this.
"""

from .control_optimization_algorithm import ControlOptimizationAlgorithm
from .cost_functional import ReducedControlCostFunctional
from .line_search import ArmijoLineSearch
from .optimal_control_problem import OptimalControlProblem

__all__ = [
    "ReducedControlCostFunctional",
    "ArmijoLineSearch",
    "ControlOptimizationAlgorithm",
    "OptimalControlProblem",
]
