.. _config_topology_optimization:

Documentation of the Config Files for Topology Optimization Problems
=================================================================

Let us take a detailed look at the config files for topology optimization problems and
discusss the corresponding parameters. Note that the settings are mostly the same as
for optimal control problems (as we deal with the problems on a fixed mesh),
so the reader is referred to :ref:`config_optimal_control` for most parameters
and here we only describe those that are not already treated there.

As in :ref:`config_optimal_control`, we refer to the `documentation of the
configparser module <https://docs.python.org/3/library/configparser.html>`_ for
a detailed description of how these config files can be structured. Moreover,
we remark that cashocs has a default behavior for almost all of these
parameters, which is triggered when they are **NOT** specified in the config file,
and we will discuss this behavior for each parameter in this tutorial. For a
summary over all parameters and their default values look at
:ref:`the end of this page <config_topology_summary>`.


.. _config_top_topology_optimization:

Section TopologyOptimization
----------------------------

Topology optimization problems contain additional parameters only in the section "TopologyOptimization".
Let us go over the parameters in that section now.

The first parameter is

.. code-block:: ini

    angle_tol = 1.0

which is the absolute tolerance (in degrees) of the angle between topological derivative and level-set function and is used as stopping criterion for the topology optimization algorithms. The default value is :ini:`angle_tol = 1.0`.

The next parameter is

.. code-block:: ini

    interpolation_scheme = volume

which describes which scheme is used to interpolate the topological derivative to the mesh nodes. Possible options are :ini:`interpolation_scheme = volume` and :ini:`interpolation_scheme = angle`, where a weighting by the cell volume or angles is used, respectively. The default is given by :ini:`interpolation_scheme = volume`. Note that the angle-weighted interpolation is not available in parallel.

Next up we have

.. code-block:: ini

    normalize_topological_derivative = False

which is a boolean flag that indicates whether the topological derivative should be normalized in an :math:`L^2` sense before
using it in the optimization algorithm or not. The default is :ini:`normalize_topological_derivative = False`.

The next parameter is given by

.. code-block:: ini

    re_normalize_levelset = False

which is a boolean flag used to indicate, whether the level set function should be re-normalized in each iteration of the optimization algorithm. The default is given by :ini:`re_normalize_levelset = False`

Next, we have the following parameter

.. code-block:: ini

    topological_derivative_is_identical = False

which is a boolean flag that is used to indicate, whether the topological derivative is equal in both parts of the domain.
As this is usually not the case, the default setting is :ini:`topological_derivative_is_identical = False`.

The next parameter is given by

.. code-block:: ini

    tol_bisection = 1e-4

which determines the absolute tolerance for the bisection approach that is used to incorporate volume constraints into the topology
optimization. The default tolerance of :ini:`tol_bisection = 1e-4` should be sufficient for most applications.

Finally, the parameter

.. code-block:: ini

    max_iter_bisection = 100

determines how many iterations of the bisection approach are carried out in the worst case. The default value for this is given by
:ini:`max_iter_bisection = 100`.


.. _config_shape_summary:

Summary
-------

Finally, an overview over all configuration parameters for topology optimization and their default values can be found
in the following.


[TopologyOptimization]
******

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`angle_tol = 1.0`
      - Stopping tolerance for the angle (in degrees)
    * - :ini:`interpolation_scheme = volume`
      - The approach used to interpolate the topological derivative. Possible options: :ini:`volume` or :ini:`angle`
    * - :ini:`normalize_topological_derivative = False`
      - Whether to normalize the topological derivative in each iteration.
    * - :ini:`re_normalize_levelset = False`
      - Whether to re-normalize the levelset function in each iteration.
    * - :ini:`topological_derivative_is_identical = False`
      - Whether the topological derivative is identical for all considered parts
    * - :ini:`tol_bisection = 1e-4`
      - Tolerance for the bisection procedure (used for incorporating volume constraints)
    * - :ini:`max_iter_bisection = 100`
      - Maximum number of iterations of the bisection procedure


