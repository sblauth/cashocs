# Copyright (C) 2020-2022 Sebastian Blauth
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

"""Type hints for cashocs."""

from typing import List, Tuple, Union

import fenics

from cashocs import _forms
from cashocs import _pde_problems
from cashocs._constraints import constraints
from cashocs._optimization import cost_functional
from cashocs._optimization import optimal_control
from cashocs._optimization import shape_optimization

OptimizationProblem = Union[
    shape_optimization.ShapeOptimizationProblem, optimal_control.OptimalControlProblem
]
GradientProblem = Union[
    _pde_problems.ShapeGradientProblem, _pde_problems.ControlGradientProblem
]
FormHandler = _forms.FormHandler
MeshTuple = Tuple[
    fenics.Mesh,
    fenics.MeshFunction,
    fenics.MeshFunction,
    fenics.Measure,
    fenics.Measure,
    fenics.Measure,
]

KspOptions = List[List[List[Union[str, int, float]]]]
Constraint = Union[constraints.EqualityConstraint, constraints.InequalityConstraint]

CostFunctional = Union[
    cost_functional.IntegralFunctional,
    cost_functional.ScalarTrackingFunctional,
    cost_functional.MinMaxFunctional,
]
