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

r"""cashocs is a shape optimization and optimal control software for python.

cashocs is based on the finite element package `FEniCS <https://fenicsproject.org>`_
and uses its high-level unified form language UFL to treat general PDE constrained
optimization problems, in particular, shape optimization and optimal control problems.

The documentation for cashocs can be found `here <https://cashocs.readthedocs.io/>`_.
"""

from cashocs._constraints.constrained_problems import ConstrainedOptimalControlProblem
from cashocs._constraints.constrained_problems import (
    ConstrainedShapeOptimizationProblem,
)
from cashocs._constraints.constraints import EqualityConstraint
from cashocs._constraints.constraints import InequalityConstraint
from cashocs._loggers import LogLevel
from cashocs._loggers import set_log_level
from cashocs._optimization import verification
from cashocs._optimization.cost_functional import IntegralFunctional
from cashocs._optimization.cost_functional import MinMaxFunctional
from cashocs._optimization.cost_functional import ScalarTrackingFunctional
from cashocs._optimization.optimal_control.optimal_control_problem import (
    OptimalControlProblem,
)
from cashocs._optimization.shape_optimization.shape_optimization_problem import (
    ShapeOptimizationProblem,
)
from cashocs._utils import create_bcs_list
from cashocs._utils import create_dirichlet_bcs
from cashocs._utils import Interpolator
from cashocs.geometry import DeformationHandler
from cashocs.geometry import import_mesh
from cashocs.geometry import interval_mesh
from cashocs.geometry import MeshQuality
from cashocs.geometry import regular_box_mesh
from cashocs.geometry import regular_mesh
from cashocs.io import create_config
from cashocs.io import load_config
from cashocs.nonlinear_solvers import damped_newton_solve
from cashocs.nonlinear_solvers import newton_solve
from cashocs.nonlinear_solvers import picard_iteration

__version__ = "1.7.6"

__all__ = [
    "import_mesh",
    "LogLevel",
    "regular_mesh",
    "regular_box_mesh",
    "DeformationHandler",
    "MeshQuality",
    "newton_solve",
    "damped_newton_solve",
    "picard_iteration",
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
    "set_log_level",
    "Interpolator",
    "IntegralFunctional",
    "ScalarTrackingFunctional",
    "MinMaxFunctional",
    "interval_mesh",
]
