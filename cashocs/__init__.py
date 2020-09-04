# Copyright (C) 2020 Sebastian Blauth
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

r"""Computational adjoint based package for PDE constrained optimization

cashocs is based on the finite element package `FEniCS <https://fenicsproject.org>`_
and uses its high-level unified form language UFL to treat general PDE constrained
optimization problems, in particular, shape optimization and optimal control problems.

Note, that we assume that you are (at least somewhat) familiar with PDE constrained
optimization and FEniCS. For a introduction to these topics, we can recommend the textbooks

- Optimal Control and general PDE constrained optimization
    - `Hinze, Pinnau, Ulbrich, and Ulbrich, Optimization with PDE Constraints <https://doi.org/10.1007/978-1-4020-8839-1>`_
    - `Tröltzsch, Optimal Control of Partial Differential Equations <https://doi.org/10.1090/gsm/112>`_
- Shape Optimization
    - `Delfour and Zolesio, Shapes and Geometries <https://doi.org/10.1137/1.9780898719826>`_
    - `Sokolowski and Zolesio, Introduction to Shape Optimization <https://doi.org/10.1007/978-3-642-58106-9>`_
- FEniCS
    - `Logg, Mardal, and Wells, Automated Solution of Differential Equations by the Finite Element Method <https://doi.org/10.1007/978-3-642-23099-8>`_

However, the :ref:`tutorial <tutorial_index>` also gives many references either to the underlying theory of PDE constrained optimization or to relevant demos and documentation of FEniCS.

"""

from . import verification
from ._optimal_control.optimal_control_problem import OptimalControlProblem
from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .geometry import import_mesh, regular_box_mesh, regular_mesh, MeshQuality
from .nonlinear_solvers import damped_newton_solve
from .utils import create_bcs_list, create_config



__all__ = ['import_mesh', 'regular_mesh', 'regular_box_mesh', 'MeshQuality',
		   'damped_newton_solve', 'OptimalControlProblem', 'ShapeOptimizationProblem',
		   'create_config', 'create_bcs_list', 'verification']
