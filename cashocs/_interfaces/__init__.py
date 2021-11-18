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

"""Interfaces for general optimization problems.

This module provides interfaces for general optimization problems. These are base
classes meant to be sublcassed for more specific problems, such as optimal control
or shape optimization
"""

from .cost_functional import ReducedCostFunctional
from .line_search import LineSearch
from .optimization_algorithm import OptimizationAlgorithm
from .optimization_problem import OptimizationProblem
from .pde_problem import PDEProblem
