"""
Created on 21/02/2020, 16.07

@author: blauths

adpack is a library based on fenics for the automated treatment of PDE constrained optimization problems.
Works for both optimal control and shape optimization.

Methods
-------
MeshGen(mesh_file)
	Imports a .xdmf mesh given in mesh_file

regular_mesh(n=10, lx=1.0, ly=1.0, lz=None)
	Creates a regular (box) mesh with n elements of [0, lx] x [0, ly] if lz is None,
	or of [0, lx] x [0, ly] x [0, lz] otherwise.

regular_box_mesh(n=10, sx=0.0, sy=0.0, sz=None, ex=1.0, ey=1.0, ez=None)
	Creates a regular box mesh with n elements, of [sx, ex] x [sy, ey] (in case sz and ez are None),
	or of [sx, ex] x [sy, ey] x [sz, ez] otherwise.

NewtonSolver(F, u, bcs, rtol=1e-10, atol=1e-10, max_iter=25, convergence_type='both', norm_type='l2', damped=True, verbose=True)
	Custom (damped) Newton Solver used to solve F==0 for the variable u, with boundary conditions bcs.

create_config(path)
	Loads the config file located at path.


Classes
-------
OptimalControlProblem
	A class representing an abstract optimal control problem.

ShapeOptimizationProblem
	A class representing an abstract shape optimization problem.
"""


from .geometry import MeshGen, regular_mesh, regular_box_mesh

from .nonlinear_solvers import NewtonSolver

from .optimal_control.optimal_control_problem import OptimalControlProblem

from .shape_optimization.shape_optimization_problem import ShapeOptimizationProblem

from .helpers import create_config
