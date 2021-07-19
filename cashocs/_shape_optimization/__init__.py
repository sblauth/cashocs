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

"""Methods and classes for shape optimization problems

"""

from .regularization import Regularization
from .shape_cost_functional import ReducedShapeCostFunctional
from .shape_line_search import ArmijoLineSearch
from .shape_optimization_algorithm import ShapeOptimizationAlgorithm


__all__ = [
    "Regularization",
    "ReducedShapeCostFunctional",
    "ArmijoLineSearch",
    "ShapeOptimizationAlgorithm",
]
