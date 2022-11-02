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

"""Optimization problems with additional (in-)equatility constraints."""

from cashocs._constraints.constrained_problems import ConstrainedOptimalControlProblem
from cashocs._constraints.constrained_problems import (
    ConstrainedShapeOptimizationProblem,
)
from cashocs._constraints.constraints import EqualityConstraint
from cashocs._constraints.constraints import InequalityConstraint

__all__ = [
    "ConstrainedOptimalControlProblem",
    "ConstrainedShapeOptimizationProblem",
    "InequalityConstraint",
    "EqualityConstraint",
]
