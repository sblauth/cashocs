r"""cestrel is an adjoint-based optimization package for python.

It can be used to treat PDE constrained optimal control and shape optimization problems
numerically in an automated fashion and is based on the finite element software fenics.

.. include:: ./documentation.md
"""

from .geometry import import_mesh, regular_mesh, regular_box_mesh, MeshQuality
from .nonlinear_solvers import damped_newton_solve
from ._optimal_control.optimal_control_problem import OptimalControlProblem
from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .utils import create_config, create_bcs_list



__all__ = ['import_mesh', 'regular_mesh', 'regular_box_mesh', 'MeshQuality', 'damped_newton_solve', 'OptimalControlProblem',
		   'ShapeOptimizationProblem', 'create_config', 'create_bcs_list']
