API Reference
=============

.. automodule:: cashocs
    :no-members:
    :no-inherited-members:

Here, we detail the (public) API of cashocs.

For a more hands-on approach, we recommend the :ref:`tutorial <tutorial_index>`, which
shows many examples from PDE constrained optimization that can be efficiently
solved with cashocs.


PDE Constrained Optimization Problems
-------------------------------------

If you are using cashocs to solve PDE constrained optimization problems, you should
use the following two classes, for either optimal control or shape optimization
problems.


OptimalControlProblem
*********************
.. autoclass:: cashocs.OptimalControlProblem

ShapeOptimizationProblem
************************
.. autoclass:: cashocs.ShapeOptimizationProblem

Additionally constrained problems
---------------------------------

ConstrainedOptimalControlProblem
********************************
.. autoclass:: cashocs.ConstrainedOptimalControlProblem

ConstrainedShapeOptimizationProblem
***********************************
.. autoclass:: cashocs.ConstrainedShapeOptimizationProblem


EqualityConstraint
******************
.. autoclass:: cashocs.EqualityConstraint

InequalityConstraint
********************
.. autoclass:: cashocs.InequalityConstraint

	
Cost Functionals
----------------

IntegralFunctional
******************
.. autoclass:: cashocs.IntegralFunctional

ScalarTrackingFunctional
************************
.. autoclass:: cashocs.ScalarTrackingFunctional
    
MinMaxFunctional
****************
.. autoclass:: cashocs.MinMaxFunctional

Command Line Interface
----------------------

For the command line interface of cashocs, we have a mesh conversion tool which
converts GMSH .msh files to .xdmf ones, which can be read with the :py:func:`import mesh
<cashocs.import_mesh>` functionality. It's usage is detailed in the following.

.. _cashocs_convert:

cashocs-convert
***************

.. argparse::
    :module: cashocs._cli._convert
    :func: _generate_parser
    :prog: cashocs-convert


MeshQuality
-----------

.. autoclass:: cashocs.MeshQuality
    :noindex:

DeformationHandler
------------------

.. autoclass:: cashocs.DeformationHandler
    :noindex:

.. autoclass:: cashocs.Interpolator

Functions
---------

The functions which are directly available in cashocs are taken from the sub-modules
:py:mod:`geometry <cashocs.geometry>`, :py:mod:`nonlinear_solvers <cashocs.nonlinear_solvers>`,
and :py:mod:`utils <cashocs.utils>`. These are functions that are likely to be used
often, so that they are directly callable via ``cashocs.function`` for any of
the functions shown below. Note, that they are repeated in the API reference for
their :ref:`respective sub-modules <sub_modules>`.

import_mesh
***********
.. autofunction:: cashocs.import_mesh

interval_mesh
*************
.. autofunction:: cashocs.interval_mesh

regular_mesh
************
.. autofunction:: cashocs.regular_mesh


regular_box_mesh
****************
.. autofunction:: cashocs.regular_box_mesh


load_config
***********
.. autofunction:: cashocs.load_config

create_dirichlet_bcs
********************
.. autofunction:: cashocs.create_dirichlet_bcs


newton_solve
************
.. autofunction:: cashocs.newton_solve

set_log_level
*************
.. autofunction:: cashocs.set_log_level


Deprecated Capabilities
-----------------------

Here, we list deprecated functions and classes of cashocs, which can still be
used, but are replaced by newer instances with more capabilities. These deprecated objects
are still maintained for compatibility reasons.

damped_newton_solve
*******************
.. autofunction:: cashocs.damped_newton_solve

create_config
*************
.. autofunction:: cashocs.create_config

create_bcs_list
***************
.. autofunction:: cashocs.create_bcs_list


.. _sub_modules:

Sub-Modules
-----------

cashocs' sub-modules include several additional classes and methods that could be
potentially useful for the user. For the corresponding API documentation, we
include the previously detailed objects, too, as to give a complete documentation
of the sub-module.

.. toctree::
   :maxdepth: 5

   sub_modules/geometry
   sub_modules/io
   sub_modules/nonlinear_solvers
