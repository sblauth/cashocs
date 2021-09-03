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

"""Some solvers for the inner (unconstrained) PDAS subproblems.

"""

from .inner_gradient_descent import InnerGradientDescent
from .inner_lbfgs import InnerLBFGS
from .inner_ncg import InnerNCG
from .inner_newton import InnerNewton


__all__ = ["InnerNCG", "InnerGradientDescent", "InnerLBFGS", "InnerNewton"]
