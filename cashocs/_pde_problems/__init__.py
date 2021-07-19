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

"""Several PDE problems for optimization (shape and optimal control).

"""

from .adjoint_problem import AdjointProblem
from .gradient_problem import GradientProblem
from .hessian_problems import HessianProblem, UnconstrainedHessianProblem
from .shape_gradient_problem import ShapeGradientProblem
from .state_problem import StateProblem


__all__ = [
    "AdjointProblem",
    "GradientProblem",
    "HessianProblem",
    "ShapeGradientProblem",
    "StateProblem",
    "UnconstrainedHessianProblem",
]
