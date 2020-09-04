API Documentation
=================

.. automodule:: cashocs



PDE Constrained Optimization Problems
-------------------------------------

If you are using cashocs to solve PDE constrained optimization problems, you should
use the following two classes, for either optimal control or shape optimization
problems.


OptimalControlProblem
*********************
.. autoclass:: cashocs.OptimalControlProblem
	:members:
	:undoc-members:
	:inherited-members:
	:show-inheritance:


ShapeOptimizationProblem
************************
.. autoclass:: cashocs.ShapeOptimizationProblem
	:members:
	:undoc-members:
	:inherited-members:
	:show-inheritance:


MeshQuality
-----------

.. autoclass:: cashocs.MeshQuality
	:members:
	:undoc-members:
	:inherited-members:
	:show-inheritance:

Functions
---------

This includes several "helper" functions, which make the definition of problems
simpler.

import_mesh
***********
.. autofunction:: cashocs.import_mesh


regular_mesh
************
.. autofunction:: cashocs.regular_mesh


regular_box_mesh
****************
.. autofunction:: cashocs.regular_box_mesh


create_config
*************
.. autofunction:: cashocs.create_config

create_bcs_list
***************
.. autofunction:: cashocs.create_bcs_list


damped_newton_solve
*******************
.. autofunction:: cashocs.damped_newton_solve



Sub-Modules
-----------

The sub-modules include several duplicates of entries already shown above. If
a method or class is important or very relevant for cashocs, it can be found
in the main module directly, and there is no need for sub-modules. However,
there may still be useful classes or functions here, for public use.

.. toctree::
   :maxdepth: 5

   sub_modules/geometry
   sub_modules/nonlinear_solvers
   sub_modules/optimization_problem
   sub_modules/utils
   sub_modules/verification