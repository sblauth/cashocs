"""package for the solution of PDE constrained optimization problems

caospy is a package based on fenics for the automated
treatment of PDE constrained optimization problems.
Works for both optimal control and shape optimization.
"""

from .geometry import MeshGen, regular_mesh, regular_box_mesh
from .nonlinear_solvers import newton_solve
from ._optimal_control.optimal_control_problem import OptimalControlProblem
from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .utilities import create_config, wrap_bcs



__all__ = ['MeshGen', 'regular_mesh', 'regular_box_mesh', 'newton_solve', 'OptimalControlProblem',
		   'ShapeOptimizationProblem', 'create_config', 'wrap_bcs']
