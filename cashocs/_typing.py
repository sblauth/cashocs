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

"""Type hints for cashocs."""

from __future__ import annotations

import fenics

from cashocs import _forms
from cashocs import _pde_problems
from cashocs._constraints import constraints
from cashocs._optimization import cost_functional
from cashocs._optimization import optimal_control
from cashocs._optimization import optimization_algorithms
from cashocs._optimization import shape_optimization
from cashocs._optimization import topology_optimization

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

OptimizationProblem = (
    shape_optimization.ShapeOptimizationProblem
    | optimal_control.OptimalControlProblem
    | topology_optimization.TopologyOptimizationProblem
)
GradientProblem = (
    _pde_problems.ShapeGradientProblem | _pde_problems.ControlGradientProblem
)
FormHandler = _forms.FormHandler
MeshTuple = tuple[
    fenics.Mesh,
    fenics.MeshFunction,
    fenics.MeshFunction,
    ufl.Measure,
    ufl.Measure,
    ufl.Measure,
]
SolutionAlgorithm = optimization_algorithms.OptimizationAlgorithm

KspOption = dict[str, int | float | str | None]
Constraint = constraints.EqualityConstraint | constraints.InequalityConstraint

CostFunctional = (
    cost_functional.IntegralFunctional
    | cost_functional.ScalarTrackingFunctional
    | cost_functional.MinMaxFunctional
    | cost_functional.Functional
)

ShapeOptimizationProblem = shape_optimization.ShapeOptimizationProblem
