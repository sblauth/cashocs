.. _config_optimal_control:

Documentation of the Config Files for Optimal Control Problems
==============================================================


Let us take a look at how the config files are structured for optimal control
problems.
The corresponding config file is :download:`config.ini
</../../demos/documented/optimal_control/poisson/config.ini>`.

First of all, the config is divided into the sections: :ref:`Mesh
<config_ocp_mesh>`, :ref:`StateSystem <config_ocp_state_system>`,
:ref:`OptimizationRoutine <config_ocp_optimization_routine>`, :ref:`AlgoLBFGS
<config_ocp_algolbfgs>`, :ref:`AlgoCG <config_ocp_algocg>`, :ref:`AlgoTNM
<config_ocp_algonewton>`, :ref:`AlgoPDAS <config_ocp_algopdas>`, and :ref:`Output <config_ocp_output>`.
These manage the settings for the mesh, the state equation of the optimization
problem, the solution algorithms, and the output, respectively. Note, that the
structure of such config files is explained in-depth in the `documentation of the
configparser module <https://docs.python.org/3/library/configparser.html>`_.
In particular, the order of the entries in each section is arbitrary.

Moreover, we remark that CASHOCS has a default behavior for almost all of these
parameters, which is triggered when they are **NOT** specified in the config file,
and we will discuss this behavior for each parameter in this tutorial.
A summary of all parameters as well as their default values
can be found at :ref:`the end of this page <config_ocp_summary>`.



.. _config_ocp_mesh:

Section Mesh
------------
The mesh section consists, for optimal control problems, only of a path to the
.xdmf version of the mesh file ::

    mesh_file = ../mesh/mesh.xdmf

This section is completely optional and can be used when importing meshes generated
with GMSH. Note, that this section can become more populated and useful
for shape optimization problems, as detailed in the
:ref:`description of their config files <config_shape_mesh>`. To convert a .msh
file to the .xdmf format, you can use the command line tool :ref:`cashocs-convert <cashocs_convert>`.



.. _config_ocp_state_system:

Section StateSystem
---------------------
The state system section is used to detail how the state and adjoint systems are
solved. This includes settings for a damped Newton method and a Picard iteration.

In the following, we go over each parameter in detail. First, we have ::

    is_linear = True

This is a boolean parameter which indicates, whether the state system
is linear. This is used to speed up some computations. Note, that the program
will always work when this is set to False, as it treats the linear problem in a
nonlinear fashion and the corresponding solver converges in one iteration. However, using
``is_linear = True``
on a nonlinear state system throws an error related to FEniCS. The default value
for this parameter is ``False``.

The next parameter is defined via ::

    newton_rtol = 1e-11

This parameter determines the relative tolerance for the Newton solver that is
used to solve a nonlinear state system. Subsequently, we can also define the
absolute tolerance for the Newton solver via ::

    newton_atol = 1e-13

Note, that the default values are ``newton_rtol = 1e-11`` and ``newton_atol = 1e-13``.

The parameter ``newton_iter``, which is defined via ::

    newton_iter = 50

controls how many iterations the Newton method is allowed to make before it
terminates. This defaults to 50.

Moreover, we have the boolean ``newton_damped`` ::

    newton_damped = True

which determines whether a damping should be used (in case this is ``True``) or not
(otherwise). This parameter defaults to ``True`` if nothing is given.

Additionally, we have the boolean parameter ``newton_inexact``, defined via ::

    newton_inexact = False

which sets up an inexact Newton method for solving nonlinear problems in case this is ``True``. The default is ``False``.

The parameter ::

    newton_verbose = False

is used to make the Newton solver's output verbose. This is disabled by default.
This concludes the settings for Newton's method.


Next up, we have the parameters controlling the Picard iteration. First, we have ::

    picard_iteration = False

This is another boolean flag, used to determine, whether the state system
shall be solved using a Picard iteration (if this is ``True``) or not
(if this is ``False``). For a single state equation (i.e. one single state
variable) both options are equivalent. The difference is only active when
considering a coupled system with multiple state variables that is coupled. The
default value for this parameter is ``False``.

The tolerances for the Picard iteration are defined via ::

    picard_rtol = 1e-10
    picard_atol = 1e-12

The first parameter determines the relative tolerance used for the Picard
iteration, in case it is enabled, and the second one determines the absolute
tolerance. Their default value are given by ``picard_rtol = 1e-10`` and
``picard_atol = 1e-12``.

The maximum number of iterations of the method can be set via ::

    picard_iter = 10

and the default value for this parameter is ``picard_iter = 50``.

The parmater ``picard_verbose`` enables verbose output of the convergence of the
Picard iteration, and is set as follows ::

    picard_verbose = False

Its default value is ``False``.




.. _config_ocp_optimization_routine:

Section OptimizationRoutine
---------------------------

The following section is used to specify general parameters for the solution
algorithm, which can be customized here. The first parameter determines the
choice of the particular algorithm, via ::

    algorithm = lbfgs

The possible choices are given by

  - ``gd`` or ``gradient_descent`` : a gradient descent method

  - ``cg``, ``conjugate_gradient``, ``ncg``, ``nonlinear_cg`` : nonlinear CG methods

  - ``lbfgs`` or ``bfgs`` : limited memory BFGS method

  - ``newton`` : a truncated Newton method

  - ``pdas`` or ``primal_dual_active_set`` : a primal dual active set method (for control constraints)

Note, that there is no default value, so that this always has to be specified by
the user.

The next line of the config file is given by ::

    rtol = 1e-4

This parameter determines the relative tolerance for the solution algorithm.
In the case where no control constraints are present, this uses the "classical"
norm of the gradient of the cost functional as measure. In case there are box
constraints present, it uses the stationarity measure (see `Kelley, Iterative Methods
for Optimization <https://doi.org/10.1137/1.9781611970920>`_ as measure.
Analogously, we also have the line ::

    atol = 0.0

This determines the absolute tolerance for the solution algorithm. The default
tolerances for the optimization algorithm are given by ``rtol = 1e-3`` and
``atol = 0.0``.

Next up, we have ::

    maximum_iterations = 100

This parameter determines the maximum number of iterations carried out by the
solution algorithm before it is terminated. It defaults to
``maximum_iterations = 100``.

The initial step size for the Armijo line search can be set via ::

    initial_stepsize = 1.0

This can have an important effect on performance of the gradient descent and nonlinear
cg methods, as they do not include a built-in scaling of the step size. The default
value is ``initial_stepsize = 1.0``.

The next paramter, ``epsilon_armijo``, is defined as follows ::

    epsilon_armijo = 1e-4

This paramter describes the parameter used in the Armijo rule to determine
sufficient decrease, via

.. math:: J(u + td) \leq J(u) + \varepsilon t \left\langle g, d \right\rangle

where u is the current optimization variable, d is the search direction, t is the
step size, and g is the current gradient. Note, that :math:`\varepsilon`
corresponds to the parameter ``epsilon_armijo``.
A value of 1e-4 is recommended and commonly used (see `Nocedal and Wright,
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_), so that
we use ``epsilon_armijo = 1e-4`` as default value.

In the following line, the parameter ``beta_armijo`` is defined ::

    beta_armijo = 2

This parameter determines the factor by the which the step size is decreased
if the Armijo condition is not satisfied, i.e., we get :math:`t = \frac{t}{\beta}`as new
step size, where :math:`\beta` corresponds to ``beta_armijo``. The default value
for this parameter is ``beta_armijo = 2.0``.

Finally, we have the parameter ``soft_exit``, which is defined as ::

    soft_exit = False

This parameter determines, whether we get a hard (``False``) or soft (``True``) exit
of the optimization routine in case it does not converge. In case of a hard exit
an Exception is raised and the script does not complete. However, it can be beneficial
to still have the subsequent code be processed, which happens in case ``soft_exit = True``.
Note, however, that in this case the returned results are **NOT** optimal,
as defined by the user input parameters. Hence, the default value is ``soft_exit = False``.


The following sections describe parameters that belong to the certain solution
algorithms.


.. _config_ocp_algolbfgs:

Section AlgoLBFGS
-----------------


For the L-BFGS method we have the following parameters. First, we have
``bfgs_memory_size``, which is set via ::

    bfgs_memory_size = 2

and determines the size of the memory of the L-BFGS method. E.g., the command
above specifies that information of the previous two iterations shall be used.
The case ``bfgs_memory_size = 0`` yields the classical gradient descent method,
whereas ``bfgs_memory_size > maximum_iterations`` gives rise to the classical
BFGS method with unlimited memory. The default behavior is ``bfgs_memory_size = 5``.

Second, we have the parameter ``use_bfgs_scaling``, that is set via ::

    use_bfgs_scaling = True

This determines, whether one should use a scaling of the initial Hessian approximation
(see `Nocedal and Wright, Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is usually very beneficial and should be kept enabled, which it is by default.

.. _config_ocp_algocg:

Section AlgoCG
--------------


The parameter ::

    cg_method = PR

determines which of the nonlinear cg methods shall be used. Available are

- ``FR`` : the Fletcher-Reeves method

- ``PR`` : the Polak-Ribiere method

- ``HS`` : the Hestenes-Stiefel method

- ``DY`` : the Dai-Yuan method

- ``HZ`` : the Hager-Zhang method

The default value for this parameter is ``cg_method = FR``.

After the definition of the particular cg method, we now have parameters determining
restart strategies for these method. First up, we have the line ::

    cg_periodic_restart = False

This parameter determines, whether the CG method should be restarted with a gradient
step periodically, which can lead to faster convergence. The amount of iterations
between restarts is then determined by ::

    cg_periodic_its = 5

In this example, the NCG method is restarted after 5 iterations. The default behavior
is given by ``cg_periodic_restart = False`` and ``cg_periodic_its = 10``. This means,
if neither of the parameters is specified, no periodic restarting takes place. If,
however, only ``cg_periodic_restart = True`` is set, the default number of iterations
before a restart will be ``cg_periodic_its = 10``, unless ``cg_periodic_its`` is
defined, too.

Another possibility to restart NCG methods is based on a relative criterion
(see `Nocedal and Wright,
Numerical Optimization, Chapter 5.2 <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is enabled via the boolean flag ::

    cg_relative_restart = False

and the corresponding relative tolerance (which should lie in :math:`(0,1)`)
is determined via ::

    cg_restart_tol = 0.5

Note, that this relative restart reinitializes the iteration with a gradient
step in case subsequent gradients are not "sufficiently" orthogonal anymore. The
default behavior is given by ``cg_relative_restart = False`` and ``cg_restart_tol = 0.25``.

.. _config_ocp_algonewton:

Section AlgoTNM
------------------

The parameters for the truncated Newton method are determined in the following.

First up, we have ::

    inner_newton = cg

which determines the Krylov method for the solution of the Newton problem. Should be one
of

- ``cg`` : A linear conjugate gradient method

- ``cr`` : A conjugate residual method

Note, that these Krylov solvers are streamlined for symmetric linear
operators, which the Hessian is (should be also positive definite for a minimizer
so that the conjugate gradient method should yield good results when initialized
not too far from the optimum). The conjugate residual does not require positive
definiteness of the operator, so that it might perform slightly better when the
initial guess is further away from the optimum. The default value is ``inner_newton = cr``.

Then, we have the following line ::

    inner_newton_tolerance = 1e-15

This determines the relative tolerance of the iterative Krylov solver for the
Hessian problem. This is set to ``inner_newton_tolerance = 1e-15`` by default.

In the final line, the paramter ``max_it_inner_newton`` is defined via ::

    max_it_inner_newton = 50

This parameter determines how many iterations of the Krylov solver are performed
before the inner iteration is terminated. Note, that the approximate solution
of the Hessian problem is used after ``max_it_inner_newton`` iterations regardless
of whether this is converged or not. This defaults to ``max_it_inner_newton = 50``.



.. _config_ocp_algopdas:

Section AlgoPDAS
----------------


Let us now take a look at the parameters for the primal dual active set method.
Its first parameter is ``inner_pdas``, which is set as follows ::

    inner_pdas = newton

This parameter determines which solution algorithm is used for the inner
(unconstrained) optimization problem in the primal dual active set method.
Can be one of

- ``gd`` or ``gradient_descent`` : A gradient descent method

- ``cg``, ``conjugate_gradient``, ``ncg``, or ``nonlinear_cg`` : A nonlinear conjugate gradient method

- ``lbfgs`` or ``bfgs`` : A limited memory BFGS method

- ``newton`` : A truncated newton method

Note, that the parameters for these inner solvers are determined via the same
interfaces used for the solution algorithms, i.e, setting ::

    [OptimizationRoutine]
    algorithm = pdas

    [AlgoLBFGS]
    bfgs_memory_size = 2

    [AlgoPDAS]
    inner_pdas = bfgs

(where we do not show additional parameters) uses the limited memory BFGS method
with memory size 2 as inner solver for the primal dual active set method.
Moreover, this parameter has to be specified by the user, there is no default value.

Moreover, we have the parameter ::

    pdas_inner_tolerance = 1e-2

This parameter determines the relative tolerance used for the inner
solution algorithms.

The maximum number of (inner) iterations for the primal dual active set method are
defined via ::

    maximum_iterations_inner_pdas = 100

and this defaults to ``maximum_iterations_inner_pdas = 50``.

Finally, we have the following line ::

    pdas_regularization_parameter = 1e-4

This determines the regularization parameter for the determination of the active and
inactive sets, and should be positive. This comes from
the interpretation as semi-smooth Newton method with Moreau Yosida regularization
of the constraints. There is no default value for this parameter, it has to be
supplied by the user.


.. _config_ocp_output:

Section Output
--------------

This section determines the behavior of CASHOCS regarding output, both in the
terminal and w.r.t. output files. The first line of this section reads ::

    verbose = True

The parameter ``verbose`` determines, whether the solution algorithm generates a verbose
output in the console, useful for monitoring its convergence. This is set to
``verbose = True`` by default.

Next up, we define the parameter ``save_results`` ::

    save_results = True

If this parameter is set to True, the history of the optimization is saved in
a .json file located in the same folder as the optimization script. This is
very useful for postprocessing the results. This defaults to ``save_results = True``.

Moreover, we define the parameter ``save_txt`` ::
	
	save_txt = True

This saves the output of the optimization, which is usually shown in the terminal,
to a .txt file, which is human-readable.

We define the parameter ``save_pvd`` in the line ::

    save_pvd = False

If ``save_pvd`` is set to True, the state variables are saved to .pvd files
in a folder named "pvd", located in the same directory as the optimization script.
These can be visualized with `Paraview <https://www.paraview.org/>`_. This parameter
defaults to ``save_pvd = False``.

The next parameter is ``save_pvd_adjoint``, which is given in the line ::

    save_pvd_adjoint = False

Analogously to the previous parameter, if ``save_pvd_adjoint`` is True, the adjoint
variables are saved to .pvd files. The default value is ``save_pvd_adjoint = False``.

The next parameter is given by ``save_pvd_gradient``, which is given in the line ::

    save_pvd_gradient = False

This boolean flag ensures that a paraview with the computed gradients is saved in ``result_dir/pvd``. The main purpose of this is for debugging.

Finally, we can specify in which directory the results should be stored with the
parameter ``result_dir``, which is given in this config file by ::

    result_dir = ./results

The path given there can be either relative or absolute. In this example, the
working directory of the python script is chosen.

Moreover, we have the parameter ``time_suffix``, which adds a suffix to the result directory based on the current time. It is controlled by the line ::

	time_suffix = False



.. _config_ocp_summary:

Summary
-------

Finally, an overview over all parameters and their default values can be found
in the following.

[Mesh]
******

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - mesh_file
      -
      - optional, see :py:func:`import_mesh <cashocs.import_mesh>`

[StateSystem]
*************

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - is_linear
      - ``False``
      - using ``True`` gives an error for nonlinear problems
    * - newton_rtol
      - ``1e-11``
      - relative tolerance for Newton's method
    * - newton_atol
      - ``1e-13``
      - absolute tolerance for Newton's method
    * - newton_iter
      - ``50``
      - maximum iterations for Newton's method
    * - newton_damped
      - ``True``
      - if ``True``, damping is enabled
    * - newton_inexact
      - ``False``
      - if ``True``, an inexact Newton's method is used
    * - newton_verbose
      - ``False``
      - ``True`` enables verbose output of Newton's method
    * - picard_iteration
      - ``False``
      - ``True`` enables Picard iteration; only has an effect for multiple
        variables
    * - picard_rtol
      - ``1e-10``
      - relative tolerance for Picard iteration
    * - picard_atol
      - ``1e-12``
      - absolute tolerance for Picard iteration
    * - picard_iter
      - ``50``
      - maximum iterations for Picard iteration
    * - picard_verbose
      - ``False``
      - ``True`` enables verbose output of Picard iteration

[OptimizationRoutine]
*********************

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - algorithm
      -
      - has to be specified by the user; see :py:meth:`solve <cashocs.OptimalControlProblem.solve>`
    * - rtol
      - ``1e-3``
      - relative tolerance for the optimization algorithm
    * - atol
      - ``0.0``
      - absolute tolerance for the optimization algorithm
    * - maximum iterations
      - ``100``
      - maximum iterations for the optimization algorithm
    * - initial_stepsize
      - ``1.0``
      - initial stepsize for the first iteration in the Armijo rule
    * - epsilon_armijo
      - ``1e-4``
      -
    * - beta_armijo
      - ``2.0``
      -
    * - soft_exit
      - ``False``
      - if ``True``, the optimization algorithm does not raise an exception if
        it did not converge


[AlgoLBFGS]
***********

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - bfgs_memory_size
      - ``5``
      - memory size of the L-BFGS method
    * - use_bfgs_scaling
      - ``True``
      - if ``True``, uses a scaled identity mapping as initial guess for the inverse Hessian


[AlgoCG]
********

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - cg_method
      - ``FR``
      - specifies which nonlinear CG method is used
    * - cg_periodic_restart
      - ``False``
      - if ``True``, enables periodic restart of NCG method
    * - cg_periodic_its
      - ``10``
      - specifies, after how many iterations the NCG method is restarted, if applicable
    * - cg_relative_restart
      - ``False``
      - if ``True``, enables restart of NCG method based on a relative criterion
    * - cg_restart_tol
      - ``0.25``
      - the tolerance of the relative restart criterion, if applicable

[AlgoTNM]
*********

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - inner_newton
      - ``cr``
      - inner iterative solver for the truncated Newton method
    * - inner_newton_tolerance
      - ``1e-15``
      - relative tolerance for the inner iterative solver
    * - max_it_inner_newton
      - ``50``
      - maximum iterations for the inner iterative solver

[AlgoPDAS]
**********

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - inner_pdas
      -
      - has to be specified; determines  the optimization algorithm used to
        solve the inner sub-problems for the PDAS method
    * - pdas_inner_tolerance
      - ``1e-2``
      - relative tolerance for the inner optimization problems
    * - maximum_iterations_inner_pdas
      - ``50``
      - maximum iterations for the inner optimization algorithm
    * - pdas_regularization_parameter
      -
      - has to be specified; needs to be positive


[Output]
********

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - verbose
      - ``True``
      - if ``True``, the history of the optimization is printed to the console
    * - save_results
      - ``True``
      - if ``True``, the history of the optimization is saved to a .json file
    * - save_txt
      - ``True``
      - if ``True``, the history of the optimization is saved to a human readable .txt file
    * - save_pvd
      - ``False``
      - if ``True``, the history of the state variables over the optimization is
        saved in .pvd files
    * - save_pvd_adjoint
      - ``False``
      - if ``True``, the history of the adjoint variables over the optimization is
        saved in .pvd files
    * - save_pvd_gradient
      - ``False``
      - if ``True``, the history of the gradient(s) over the optimization is saved in .pvd files
    * - result_dir
      - ``./``
      - path to the directory, where the output should be placed
    * - time_suffix
      - ``False``
      - Boolean flag, which adds a suffix to ``result_dir`` based on the current time


This concludes the documentation of the config files for optimal control problems.
For the corresponding documentation for shape optimization problems, see :ref:`config_shape_optimization`.
