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

r"""CASHOCS is a computational, adjoint based shape optimization and optimal control software for python.

CASHOCS is based on the finite element package `FEniCS <https://fenicsproject.org>`_
and uses its high-level unified form language UFL to treat general PDE constrained
optimization problems, in particular, shape optimization and optimal control problems.
"""

from . import verification
from ._constraints.constrained_problems import (
    ConstrainedOptimalControlProblem,
    ConstrainedShapeOptimizationProblem,
)
from ._constraints.constraints import EqualityConstraint, InequalityConstraint
from ._loggers import LogLevel, set_log_level
from ._optimal_control.optimal_control_problem import OptimalControlProblem
from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .geometry import (
    DeformationHandler,
    MeshQuality,
    import_mesh,
    regular_box_mesh,
    regular_mesh,
)
from .nonlinear_solvers import newton_solve, damped_newton_solve
from .utils import (
    create_bcs_list,
    create_dirichlet_bcs,
    create_config,
    load_config,
)


__version__ = "1.4.8"

__all__ = [
    "import_mesh",
    "regular_mesh",
    "regular_box_mesh",
    "DeformationHandler",
    "MeshQuality",
    "newton_solve",
    "OptimalControlProblem",
    "ShapeOptimizationProblem",
    "create_config",
    "load_config",
    "create_bcs_list",
    "create_dirichlet_bcs",
    "verification",
    "ConstrainedOptimalControlProblem",
    "ConstrainedShapeOptimizationProblem",
    "EqualityConstraint",
    "InequalityConstraint",
]
