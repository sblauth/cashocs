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

r"""cashocs is a shape optimization and optimal control software for python.

cashocs is based on the finite element package `FEniCS <https://fenicsproject.org>`_
and uses its high-level unified form language UFL to treat general PDE constrained
optimization problems, in particular, shape optimization and optimal control problems.

The documentation for cashocs can be found `here <https://cashocs.readthedocs.io/>`_.
"""

from cashocs import geometry
from cashocs import io
from cashocs import log
from cashocs import mpi
from cashocs import nonlinear_solvers
from cashocs import space_mapping
from cashocs import verification
from cashocs._constraints.constrained_problems import ConstrainedOptimalControlProblem
from cashocs._constraints.constrained_problems import (
    ConstrainedShapeOptimizationProblem,
)
from cashocs._constraints.constraints import EqualityConstraint
from cashocs._constraints.constraints import InequalityConstraint
from cashocs._optimization.cost_functional import Functional
from cashocs._optimization.cost_functional import IntegralFunctional
from cashocs._optimization.cost_functional import MinMaxFunctional
from cashocs._optimization.cost_functional import ScalarTrackingFunctional
from cashocs._optimization.optimal_control.optimal_control_problem import (
    OptimalControlProblem,
)
from cashocs._optimization.shape_optimization.shape_optimization_problem import (
    ShapeOptimizationProblem,
)
from cashocs._optimization.topology_optimization import TopologyOptimizationProblem
from cashocs._utils import create_dirichlet_bcs
from cashocs._utils import interpolate_levelset_function_to_cells
from cashocs._utils import Interpolator
from cashocs.geometry import compute_mesh_quality
from cashocs.geometry import interval_mesh
from cashocs.geometry import regular_box_mesh
from cashocs.geometry import regular_mesh
from cashocs.io import convert
from cashocs.io import import_mesh
from cashocs.io import load_config
from cashocs.log import LogLevel
from cashocs.log import set_log_level
from cashocs.nonlinear_solvers import linear_solve
from cashocs.nonlinear_solvers import newton_solve
from cashocs.nonlinear_solvers import picard_iteration
from cashocs.nonlinear_solvers import snes_solve
from cashocs.nonlinear_solvers import ts_pseudo_solve

__version__ = "2.7.3"

__citation__ = """
@Article{Blauth2021cashocs,
  author   = {Sebastian Blauth},
  journal  = {SoftwareX},
  title    = {{cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal
  Control Software}},
  year     = {2021},
  issn     = {2352-7110},
  pages    = {100646},
  volume   = {13},
  doi      = {https://doi.org/10.1016/j.softx.2020.100646},
}

@Misc{Blauth2023Version,
  author        = {Sebastian Blauth},
  title         = {{Version 2.0 -- cashocs: A Computational, Adjoint-Based Shape
  Optimization and Optimal Control Software}},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2306.09828},
  primaryclass  = {math.OC},
}

@Article{Blauth2021Nonlinear,
  author     = {Blauth, Sebastian},
  journal    = {SIAM J. Optim.},
  title      = {{Nonlinear Conjugate Gradient Methods for PDE Constrained Shape
  Optimization Based on Steklov-Poincar\'{e}-Type Metrics}},
  year       = {2021},
  issn       = {1052-6234,1095-7189},
  number     = {3},
  pages      = {1658--1689},
  volume     = {31},
  doi        = {10.1137/20M1367738},
  fjournal   = {SIAM Journal on Optimization},
  groups     = {My Publications, Shape Optimization},
  mrclass    = {49Q10 (35Q93 49M05 49M37 90C53)},
  mrnumber   = {4281312},
}

@Article{Blauth2023Space,
  author   = {Blauth, Sebastian},
  journal  = {SIAM J. Optim.},
  title    = {Space {M}apping for {PDE} {C}onstrained {S}hape {O}ptimization},
  year     = {2023},
  issn     = {1052-6234,1095-7189},
  number   = {3},
  pages    = {1707--1733},
  volume   = {33},
  doi      = {10.1137/22M1515665},
  fjournal = {SIAM Journal on Optimization},
  mrclass  = {49Q10 (35Q93 49M41 65K05)},
  mrnumber = {4622415},
}

@article{Blauth2023Quasi,
  author        = {Sebastian Blauth and Kevin Sturm},
  title         = {{Quasi-Newton Methods for Topology Optimization Using a Level-Set
  Method}},
  year          = {2023},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2303.15070},
}
"""

__all__ = [
    "geometry",
    "io",
    "log",
    "mpi",
    "nonlinear_solvers",
    "space_mapping",
    "verification",
    "ConstrainedOptimalControlProblem",
    "ConstrainedShapeOptimizationProblem",
    "EqualityConstraint",
    "InequalityConstraint",
    "Functional",
    "IntegralFunctional",
    "MinMaxFunctional",
    "ScalarTrackingFunctional",
    "OptimalControlProblem",
    "ShapeOptimizationProblem",
    "TopologyOptimizationProblem",
    "create_dirichlet_bcs",
    "interpolate_levelset_function_to_cells",
    "Interpolator",
    "compute_mesh_quality",
    "interval_mesh",
    "regular_box_mesh",
    "regular_mesh",
    "convert",
    "import_mesh",
    "load_config",
    "LogLevel",
    "set_log_level",
    "linear_solve",
    "newton_solve",
    "picard_iteration",
    "snes_solve",
    "ts_pseudo_solve",
]
