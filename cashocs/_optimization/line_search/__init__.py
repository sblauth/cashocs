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

"""Line search algorithms."""

from cashocs._optimization.line_search.armijo_line_search import ArmijoLineSearch
from cashocs._optimization.line_search.line_search import LineSearch
from cashocs._optimization.line_search.polynomial_line_search import (
    PolynomialLineSearch,
)

__all__ = [
    "ArmijoLineSearch",
    "LineSearch",
    "PolynomialLineSearch",
]
