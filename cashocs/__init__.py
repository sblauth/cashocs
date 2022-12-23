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

from cashocs import space_mapping
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
from cashocs._utils import create_dirichlet_bcs
from cashocs._utils import Interpolator
from cashocs.geometry import compute_mesh_quality
from cashocs.geometry import DeformationHandler
from cashocs.geometry import interval_mesh
from cashocs.geometry import regular_box_mesh
from cashocs.geometry import regular_mesh
from cashocs.io import convert
from cashocs.io import import_mesh
from cashocs.io import load_config
from cashocs.nonlinear_solvers import newton_solve
from cashocs.nonlinear_solvers import picard_iteration

__version__ = "2.0.0-alpha1"

__citation__ = """
@Article{Blauth2021cashocs,
    author   = {Sebastian Blauth},
    journal  = {SoftwareX},
    title    = {{cashocs: A Computational, Adjoint-Based Shape Optimization and
        Optimal Control Software}},
    year     = {2021},
    issn     = {2352-7110},
    pages    = {100646},
    volume   = {13},
    doi      = {https://doi.org/10.1016/j.softx.2020.100646},
    keywords = {PDE constrained optimization, Adjoint approach, Shape optimization,
        Optimal control},
}

@Article{Blauth2021Nonlinear,
    author   = {Sebastian Blauth},
    journal  = {SIAM J. Optim.},
    title    = {{N}onlinear {C}onjugate {G}radient {M}ethods for {PDE} {C}onstrained
        {S}hape {O}ptimization {B}ased on {S}teklov-{P}oincaré-{T}ype {M}etrics},
    year     = {2021},
    number   = {3},
    pages    = {1658--1689},
    volume   = {31},
    doi      = {10.1137/20M1367738},
    fjournal = {SIAM Journal on Optimization},
}
"""

__all__ = [
    "import_mesh",
    "LogLevel",
    "regular_mesh",
    "regular_box_mesh",
    "DeformationHandler",
    "compute_mesh_quality",
    "newton_solve",
    "picard_iteration",
    "OptimalControlProblem",
    "ShapeOptimizationProblem",
    "load_config",
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
    "convert",
    "space_mapping",
]
