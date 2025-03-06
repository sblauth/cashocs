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

"""Custom solvers for nonlinear equations.

This module has custom solvers for nonlinear PDEs, including a damped Newton method and
a Picard iteration for coupled problems.
"""

from cashocs.nonlinear_solvers import linear_solver
from cashocs.nonlinear_solvers import newton_solver
from cashocs.nonlinear_solvers import picard_solver
from cashocs.nonlinear_solvers import snes
from cashocs.nonlinear_solvers import ts
from cashocs.nonlinear_solvers.linear_solver import linear_solve
from cashocs.nonlinear_solvers.newton_solver import newton_solve
from cashocs.nonlinear_solvers.picard_solver import picard_iteration
from cashocs.nonlinear_solvers.snes import snes_solve
from cashocs.nonlinear_solvers.ts import ts_pseudo_solve

__all__ = [
    "linear_solver",
    "newton_solver",
    "picard_solver",
    "snes",
    "ts",
    "linear_solve",
    "newton_solve",
    "picard_iteration",
    "snes_solve",
    "ts_pseudo_solve",
]
