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

"""PDEs associated to optimization problems."""

from cashocs._pde_problems.adjoint_problem import AdjointProblem
from cashocs._pde_problems.control_gradient_problem import ControlGradientProblem
from cashocs._pde_problems.hessian_problems import HessianProblem
from cashocs._pde_problems.pde_problem import PDEProblem
from cashocs._pde_problems.shape_gradient_problem import ShapeGradientProblem
from cashocs._pde_problems.state_problem import StateProblem

__all__ = [
    "AdjointProblem",
    "ControlGradientProblem",
    "HessianProblem",
    "PDEProblem",
    "ShapeGradientProblem",
    "StateProblem",
]
