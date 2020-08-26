r"""cashocs is a Computational Adjoint based SHape optimization and Optimal Control Software for python.

cashocs can be used to treat optimal control and shape optimization
problems constrained by PDEs. It derives the necessary adjoint
equations automatically and implements various solvers for the
problems. cashocs is based in the finite element fenics and
allows the user to define the optimization problems in the
high-level unified form language (UFL) of fenics.

.. include:: ./documentation.md

"""

from .geometry import import_mesh, regular_mesh, regular_box_mesh, MeshQuality
from .nonlinear_solvers import damped_newton_solve
from ._optimal_control.optimal_control_problem import OptimalControlProblem
from ._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .utils import create_config, create_bcs_list
from . import verification


__all__ = ['import_mesh', 'regular_mesh', 'regular_box_mesh', 'MeshQuality',
		   'damped_newton_solve', 'OptimalControlProblem', 'ShapeOptimizationProblem',
		   'create_config', 'create_bcs_list', 'verification']
