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

"""Optimization algorithms for optimal control problems

In this module, the optimization algorithms for the optimal control
problems are implemented.
"""

from .gradient_descent import GradientDescent
from .l_bfgs import LBFGS
from .ncg import NCG
from .newton import Newton
from .pdas import PDAS


__all__ = ["GradientDescent", "NCG", "LBFGS", "Newton", "PDAS"]
