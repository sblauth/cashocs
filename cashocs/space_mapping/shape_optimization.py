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

"""Space mapping for shape optimization problems.

"""

import numpy as np
import fenics
from ufl import replace
from _collections import deque


class ParentFineModel:
    def __init__(self):
        self.control = None
        self.cost_functional_value = None

    def solve_and_evaluate(self):
        pass


class CoarseModel:
    def __init__(self):
        pass

    def optimize(self):
        pass


class ParameterExtraction:
    def __init__(self):
        pass

    def _solve(self):
        pass


class SpaceMapping:
    def __init__(self):
        pass

    def solve(self):
        pass

    def _inner_product(self):
        pass

    def _compute_search_direction(self):
        pass

    def _compute_steepest_descent_application(self):
        pass

    def _compute_broyden_application(self):
        pass

    def _compute_bfgs_application(self):
        pass

    def _compute_ncg_direction(self):
        pass

    def _compute_eps(self):
        pass
