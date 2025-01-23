.. _config_optimal_control:

Documentation of the Config Files for Optimal Control Problems
==============================================================


Let us take a look at how the config files are structured for optimal control
problems.

First of all, the config is divided into the sections: :ref:`Mesh
<config_ocp_mesh>`, :ref:`StateSystem <config_ocp_state_system>`,
:ref:`OptimizationRoutine <config_ocp_optimization_routine>`, :ref:`AlgoLBFGS
<config_ocp_algolbfgs>`, :ref:`AlgoCG <config_ocp_algocg>`, :ref:`AlgoTNM
<config_ocp_algonewton>`, and :ref:`Output <config_ocp_output>`.
These manage the settings for the mesh, the state equation of the optimization
problem, the solution algorithms, and the output, respectively. Note, that the
structure of such config files is explained in-depth in the `documentation of the
configparser module <https://docs.python.org/3/library/configparser.html>`_.
In particular, the order of the entries in each section is arbitrary.

Moreover, we remark that cashocs has a default behavior for almost all of these
parameters, which is triggered when they are **NOT** specified in the config file,
and we will discuss this behavior for each parameter in this tutorial.
A summary of all parameters as well as their default values
can be found at :ref:`the end of this page <config_ocp_summary>`.



.. _config_ocp_mesh:

Section Mesh
------------
The mesh section consists, for optimal control problems, only of a path to the
.xdmf version of the mesh file

.. code-block:: ini

    mesh_file = ../mesh/mesh.xdmf

This section is completely optional and can be used when importing meshes generated
with GMSH. Note, that this section can become more populated and useful
for shape optimization problems, as detailed in the
:ref:`description of their config files <config_shape_mesh>`. To convert a .msh
file to the .xdmf format, you can use :py:func:`cashocs.convert`.



.. _config_ocp_state_system:

Section StateSystem
---------------------
The state system section is used to detail how the state and adjoint systems are
solved. This includes settings for a damped Newton method and a Picard iteration.

In the following, we go over each parameter in detail. First, we have 

.. code-block:: ini

    is_linear = True

This is a boolean parameter which indicates, whether the state system
is linear. This is used to speed up some computations. Note, that the program
will always work when this is set to False, as it treats the linear problem in a
nonlinear fashion and the corresponding solver converges in one iteration. However, using
:ini:`is_linear = True`
on a nonlinear state system throws an error related to FEniCS. The default value
for this parameter is :ini:`is_linear = False`.

The next parameter is defined via 

.. code-block:: ini

    newton_rtol = 1e-11

This parameter determines the relative tolerance for the Newton solver that is
used to solve a nonlinear state system. Subsequently, we can also define the
absolute tolerance for the Newton solver via 

.. code-block:: ini

    newton_atol = 1e-13

Note, that the default values are :ini:`newton_rtol = 1e-11` and :ini:`newton_atol = 1e-13`.

The parameter :ini:`newton_iter`, which is defined via

.. code-block:: ini

    newton_iter = 50

controls how many iterations the Newton method is allowed to make before it
terminates. This defaults to :ini:`newton_iter = 50`.

Moreover, we have the boolean :ini:`newton_damped` 

.. code-block:: ini

    newton_damped = True

which determines whether a damping should be used (in case this is :ini:`newton_damped = True`) or not
(otherwise). This parameter defaults to :ini:`newton_damped = False` if nothing is given.

Additionally, we have the boolean parameter :ini:`newton_inexact`, defined via 

.. code-block:: ini

    newton_inexact = False

which sets up an inexact Newton method for solving nonlinear problems in case this is :ini:`newton_inexact = True`. The default is :ini:`newton_inexact = False`.

The parameter 

.. code-block:: ini

    newton_verbose = False

is used to make the Newton solver's output verbose. This is disabled by default.
This concludes the settings for Newton's method.


Next up, we have the parameters controlling the Picard iteration. First, we have 

.. code-block:: ini

    picard_iteration = False

This is another boolean flag, used to determine, whether the state system
shall be solved using a Picard iteration (if this is :ini:`picard_iteration = True`) or not
(if this is :ini:`picard_iteration = False`). For a single state equation (i.e. one single state
variable) both options are equivalent. The difference is only active when
considering a coupled system with multiple state variables that is coupled. The
default value for this parameter is :ini:`picard_iteration = False`.

The tolerances for the Picard iteration are defined via 

.. code-block:: ini

    picard_rtol = 1e-10
    picard_atol = 1e-12

The first parameter determines the relative tolerance used for the Picard
iteration, in case it is enabled, and the second one determines the absolute
tolerance. Their default value are given by :ini:`picard_rtol = 1e-10` and
:ini:`picard_atol = 1e-12`.

The maximum number of iterations of the method can be set via 

.. code-block:: ini

    picard_iter = 10

and the default value for this parameter is :ini:`picard_iter = 50`.

The parmater :ini:`picard_verbose` enables verbose output of the convergence of the
Picard iteration, and is set as follows 

.. code-block:: ini

    picard_verbose = False

Its default value is :ini:`False`.

The parameter :ini:`backend` specifies which solver backend should be used for solving nonlinear systems.
Its default value is given by

.. code-block:: ini

    backend = cashocs

Possible options are :ini:`backend = cashocs` and :ini:`backend = petsc`. In the former case, a 
damped, inexact Newton method which is affine co-variant is used. Its parameters are specified in the
configuration above. In the latter case, PETSc's SNES interface for solving nonlinear equations
is used which can be configured with the `ksp_options` supplied by the user to the 
:py:class:`cashocs.OptimizationProblem`. An overview over possible PETSc command line options
can be found at `<https://petsc.org/release/manualpages/SNES/>`_.




.. _config_ocp_optimization_routine:

Section OptimizationRoutine
---------------------------

The following section is used to specify general parameters for the solution
algorithm, which can be customized here. The first parameter determines the
choice of the particular algorithm, via 

.. code-block:: ini

    algorithm = lbfgs

The possible choices are given by

  - :ini:`algorithm = gd` or :ini:`algorithm = gradient_descent` : a gradient descent method

  - :ini:`algorithm = cg`, :ini:`algorithm = conjugate_gradient`, :ini:`algorithm = ncg`, :ini:`algorithm = nonlinear_cg` : nonlinear CG methods

  - :ini:`algorithm = lbfgs` or :ini:`algorithm = bfgs` : limited memory BFGS method

  - :ini:`algorithm = newton` : a truncated Newton method

Note, that there is no default value, so that this always has to be specified by
the user.

The next line of the config file is given by 

.. code-block:: ini

    rtol = 1e-4

This parameter determines the relative tolerance for the solution algorithm.
In the case where no control constraints are present, this uses the "classical"
norm of the gradient of the cost functional as measure. In case there are box
constraints present, it uses the stationarity measure (see `Kelley, Iterative Methods
for Optimization <https://doi.org/10.1137/1.9781611970920>`_ as measure.
Analogously, we also have the line 

.. code-block:: ini

    atol = 0.0

This determines the absolute tolerance for the solution algorithm. The default
tolerances for the optimization algorithm are given by :ini:`rtol = 1e-3` and
:ini:`atol = 0.0`.

Next up, we have 

.. code-block:: ini

    max_iter = 100

This parameter determines the maximum number of iterations carried out by the
solution algorithm before it is terminated. It defaults to
:ini:`max_iter = 100`.

The initial step size for the Armijo line search can be set via 

.. code-block:: ini

    initial_stepsize = 1.0

This can have an important effect on performance of the gradient descent and nonlinear
cg methods, as they do not include a built-in scaling of the step size. The default
value is :ini:`initial_stepsize = 1.0`.

The next parameter is given by 

.. code-block:: ini

    safeguard_stepsize = True
    
This parameter can be used to activate safeguarding of the initial stepsize for line search methods. This helps
to choose an apropriate stepsize for the initial iteration even if the problem is poorly scaled. 

The next paramter, :ini:`epsilon_armijo`, is defined as follows 

.. code-block:: ini

    epsilon_armijo = 1e-4

This paramter describes the parameter used in the Armijo rule to determine
sufficient decrease, via

.. math:: J(u + td) \leq J(u) + \varepsilon t \left\langle g, d \right\rangle

where u is the current optimization variable, d is the search direction, t is the
step size, and g is the current gradient. Note, that :math:`\varepsilon`
corresponds to the parameter :ini:`epsilon_armijo`.
A value of 1e-4 is recommended and commonly used (see `Nocedal and Wright -
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_), so that
we use :ini:`epsilon_armijo = 1e-4` as default value.

In the following line, the parameter :ini:`beta_armijo` is defined 

.. code-block:: ini

    beta_armijo = 2

This parameter determines the factor by the which the step size is decreased
if the Armijo condition is not satisfied, i.e., we get :math:`t = \frac{t}{\beta}`as new
step size, where :math:`\beta` corresponds to :ini:`beta_armijo`. The default value
for this parameter is :ini:`beta_armijo = 2.0`.

Next, we have a set of two parameters which detail the methods used for computing gradients in cashocs.
These parameters are 

.. code-block:: ini

    gradient_method = direct
    
as well as 

.. code-block:: ini

    gradient_tol = 1e-9

The first parameter, :ini:`gradient_method` can be either :ini:`gradient_method = direct` or :ini:`gradient_method = iterative`. In the former case, a
direct solver is used to compute the gradient (using a Riesz projection) and in the latter case, an
iterative solver is used to do so. In case we have :ini:`gradient_method = iterative`, the parameter 
:ini:`gradient_tol` is used to specify the (relative) tolerance for the iterative solver, in the other case 
the parameter is not used.

Finally, we have the parameter :ini:`soft_exit`, which is defined as 

.. code-block:: ini

    soft_exit = False

This parameter determines, whether we get a hard (:ini:`soft_exit = False`) or soft (:ini:`soft_exit = True`) exit
of the optimization routine in case it does not converge. In case of a hard exit
an Exception is raised and the script does not complete. However, it can be beneficial
to still have the subsequent code be processed, which happens in case :ini:`soft_exit = True`.
Note, however, that in this case the returned results are **NOT** optimal,
as defined by the user input parameters. Hence, the default value is :ini:`soft_exit = False`.


The following sections describe parameters that belong to the certain solution
algorithms.


.. _config_ocp_linesearch:

Section LineSearch
------------------

In this section, parameters regarding the line search can be specified. The type of the line search can be chosen via the parameter 

.. code-block:: ini

    method = armijo
    
Possible options are :ini:`method = armijo`, which performs a simple backtracking line search based on the armijo rule with fixed steps (think of halving the stepsize in each iteration), and :ini:`method = polynomial`, which uses polynomial models of the cost functional restricted to the line to generate "better" guesses for the stepsize. The default is :ini:`method = armijo`. 

The next parameter, :ini:`polynomial_model`, specifies, which type of polynomials are used to generate new trial stepsizes. It is set via 

.. code-block:: ini

    polynomial_model = cubic
    
The parameter can either be :ini:`polynomial_model = quadratic` or :ini:`polynomial_model = cubic`. If this is :ini:`polynomial_model = quadratic`, a quadratic interpolation polynomial along the search direction is generated and this is minimized analytically to generate a new trial stepsize. Here, only the current function value, the direction derivative of the cost functional in direction of the search direction, and the most recent trial stepsize are used to generate the polynomial. In case that :ini:`polynomial_model = cubic`, the last two trial stepsizes (when available) are used in addition to the current cost functional value and the directional derivative, to generate a cubic model of the one-dimensional cost functional, which is then minimized to compute a new trial stepsize.

For the polynomial models, we also have a safeguarding procedure, which ensures that trial stepsizes cannot be chosen too large or too small, and which can be configured with the following two parameters. The trial stepsizes generate by the polynomial models are projected to the interval :math:`[\beta_{low} \alpha, \beta_{high} \alpha]`, where :math:`\alpha` is the previous trial stepsize and :math:`\beta_{low}, \beta_{high}` are factors which can be set via the parameters :ini:`factor_low` and :ini:`factor_high`. In the config file, this can look like this 

.. code-block:: ini

    factor_high = 0.5
    factor_low = 0.1

and the values specified here are also the default values for these parameters.

Finally, we have the parameter

.. code-block:: ini

    fail_if_not_converged = False

which determines, whether the line search is terminated if the state system cannot be solved at the current iterate. If this is :ini:`fail_if_not_converged = True`, then an exception is raised. Otherwise, the iterate is counted as having too high of a function value and the stepsize is "halved" and a new iterate is formed.

.. _config_ocp_algolbfgs:

Section AlgoLBFGS
-----------------


For the L-BFGS method we have the following parameters. First, we have
:ini:`bfgs_memory_size`, which is set via 

.. code-block:: ini

    bfgs_memory_size = 2

and determines the size of the memory of the L-BFGS method. E.g., the command
above specifies that information of the previous two iterations shall be used.
The case :ini:`bfgs_memory_size = 0` yields the classical gradient descent method,
whereas :python:`bfgs_memory_size > max_iter` gives rise to the classical
BFGS method with unlimited memory. The default behavior is :ini:`bfgs_memory_size = 5`.

Second, we have the parameter :ini:`use_bfgs_scaling`, that is set via 

.. code-block:: ini

    use_bfgs_scaling = True

This determines, whether one should use a scaling of the initial Hessian approximation
(see `Nocedal and Wright - Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is usually very beneficial and should be kept enabled, which it is by default.

Third, we have the parameter :ini:`bfgs_periodic_restart`, which is set in the line 

.. code-block:: ini

    bfgs_periodic_restart = 0
   
This is a non-negative integer value, which indicates the number of BFGS iterations, before a reinitialization takes place. In case that this is :ini:`bfgs_periodic_restart = 0` (which is the default), no restarts are performed. 

Finally, we have the parameter :ini:`damped`, which can be set with

.. code-block:: ini

    damped = False

This parameter is a boolean flag, which indicates whether Powell's damping (on H) should be used or not. This is useful, when the curvature condition is not satisfied and (without damping) a restart would be required. The default is :ini:`damped = False`.

.. _config_ocp_algocg:

Section AlgoCG
--------------


The parameter 

.. code-block:: ini

    cg_method = PR

determines which of the nonlinear cg methods shall be used. Available are

- :ini:`cg_method = FR` : the Fletcher-Reeves method

- :ini:`cg_method = PR` : the Polak-Ribiere method

- :ini:`cg_method = HS` : the Hestenes-Stiefel method

- :ini:`cg_method = DY` : the Dai-Yuan method

- :ini:`cg_method = HZ` : the Hager-Zhang method

The default value for this parameter is :ini:`cg_method = FR`.

After the definition of the particular cg method, we now have parameters determining
restart strategies for these method. First up, we have the line 

.. code-block:: ini

    cg_periodic_restart = False

This parameter determines, whether the CG method should be restarted with a gradient
step periodically, which can lead to faster convergence. The amount of iterations
between restarts is then determined by 

.. code-block:: ini

    cg_periodic_its = 5

In this example, the NCG method is restarted after 5 iterations. The default behavior
is given by :ini:`cg_periodic_restart = False` and :ini:`cg_periodic_its = 10`. This means,
if neither of the parameters is specified, no periodic restarting takes place. If,
however, only :ini:`cg_periodic_restart = True` is set, the default number of iterations
before a restart will be :ini:`cg_periodic_its = 10`, unless :ini:`cg_periodic_its` is
defined, too.

Another possibility to restart NCG methods is based on a relative criterion
(see `Nocedal and Wright -
Numerical Optimization, Chapter 5.2 <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is enabled via the boolean flag 

.. code-block:: ini

    cg_relative_restart = False

and the corresponding relative tolerance (which should lie in :math:`(0,1)`)
is determined via 

.. code-block:: ini

    cg_restart_tol = 0.5

Note, that this relative restart reinitializes the iteration with a gradient
step in case subsequent gradients are not "sufficiently" orthogonal anymore. The
default behavior is given by :ini:`cg_relative_restart = False` and :ini:`cg_restart_tol = 0.25`.

.. _config_ocp_algonewton:

Section AlgoTNM
------------------

The parameters for the truncated Newton method are determined in the following.

First up, we have 

.. code-block:: ini

    inner_newton = cg

which determines the Krylov method for the solution of the Newton problem. Should be one
of

- :ini:`inner_newton = cg` : A linear conjugate gradient method

- :ini:`inner_newton = cr` : A conjugate residual method

Note, that these Krylov solvers are streamlined for symmetric linear
operators, which the Hessian is (should be also positive definite for a minimizer
so that the conjugate gradient method should yield good results when initialized
not too far from the optimum). The conjugate residual does not require positive
definiteness of the operator, so that it might perform slightly better when the
initial guess is further away from the optimum. The default value is :ini:`inner_newton = cr`.

Then, we have the following line 

.. code-block:: ini

    inner_newton_rtol = 1e-15

This determines the relative tolerance of the iterative Krylov solver for the
Hessian problem. This is set to :ini:`inner_newton_rtol = 1e-15` by default.

Moreover, we can also specify the absolute tolerance for the iterative solver for the
Hessian problem, with the line 

.. code-block:: ini

    inner_newton_atol = 1e-15

analogously to the relative tolerance above.

In the final line, the paramter :ini:`max_it_inner_newton` is defined via 

.. code-block:: ini

    max_it_inner_newton = 50

This parameter determines how many iterations of the Krylov solver are performed
before the inner iteration is terminated. Note, that the approximate solution
of the Hessian problem is used after :ini:`max_it_inner_newton` iterations regardless
of whether this is converged or not. This defaults to :ini:`max_it_inner_newton = 50`.


.. _config_ocp_output:

Section Output
--------------

This section determines the behavior of cashocs regarding output, both in the
terminal and w.r.t. output files. The first line of this section reads

.. code-block:: ini

    verbose = True

The parameter :ini:`verbose` determines, whether the solution algorithm generates a verbose
output in the console, useful for monitoring its convergence. This is set to
:ini:`verbose = True` by default.

Next up, we define the parameter :ini:`save_results` 

.. code-block:: ini

    save_results = True

If this parameter is set to True, the history of the optimization is saved in
a .json file located in the same folder as the optimization script. This is
very useful for postprocessing the results. This defaults to :ini:`save_results = True`.

Moreover, we define the parameter :ini:`save_txt` 

.. code-block:: ini
	
	save_txt = True

This saves the output of the optimization, which is usually shown in the terminal,
to a .txt file, which is human-readable.

We define the parameter :ini:`save_state` in the line 

.. code-block:: ini

    save_state = False

If :ini:`save_state = True`, the state variables are saved to .xdmf files
in a folder named "xdmf", located in the same directory as the optimization script.
These can be visualized with `Paraview <https://www.paraview.org/>`_. This parameter
defaults to :ini:`save_state = False`.

The next parameter is :ini:`save_adjoint`, which is given in the line 

.. code-block:: ini

    save_adjoint = False

Analogously to the previous parameter, if :ini:`save_adjoint = True`, the adjoint
variables are saved to .xdmf files. The default value is :ini:`save_adjoint = False`.

The next parameter is given by :ini:`save_gradient`, which is given in the line 

.. code-block:: ini

    save_gradient = False

This boolean flag ensures that a paraview with the computed gradients is saved in `result_dir/xdmf`. The main purpose of this is for debugging.

Finally, we can specify in which directory the results should be stored with the
parameter :ini:`result_dir`, which is given in this config file by 

.. code-block:: ini

    result_dir = ./results

The path given there can be either relative or absolute. In this example, the
working directory of the python script is chosen.

The parameter :ini:`precision`, which is set via 

.. code-block:: ini

    precision = 3

is an integer parameter which determines how many significant digits are printed in the output to the console and / or the result file.

Moreover, we have the parameter :ini:`time_suffix`, which adds a suffix to the result directory based on the current time. It is controlled by the line 

.. code-block:: ini

	time_suffix = False



.. _config_ocp_summary:

Summary
-------

Finally, an overview over all parameters and their default values can be found
in the following.


[StateSystem]
*************

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`is_linear = False`
      - using :ini:`is_linear = True` gives an error for nonlinear problems
    * - :ini:`newton_rtol = 1e-11`
      - relative tolerance for Newton's method
    * - :ini:`newton_atol = 1e-13`
      - absolute tolerance for Newton's method
    * - :ini:`newton_iter = 50`
      - maximum iterations for Newton's method
    * - :ini:`newton_damped = False`
      - if :ini:`newton_damped = True`, damping is enabled
    * - :ini:`newton_inexact = False`
      - if :ini:`newton_inexact = True`, an inexact Newton's method is used
    * - :ini:`newton_verbose = False`
      - :ini:`newton_verbose = True` enables verbose output of Newton's method
    * - :ini:`picard_iteration = False`
      - :ini:`picard_iteration = True` enables Picard iteration; only has an effect for multiple
        variables
    * - :ini:`picard_rtol = 1e-10`
      - relative tolerance for Picard iteration
    * - :ini:`picard_atol = 1e-12`
      - absolute tolerance for Picard iteration
    * - :ini:`picard_iter = 50`
      - maximum iterations for Picard iteration
    * - :ini:`picard_verbose = False`
      - :ini:`picard_verbose = True` enables verbose output of Picard iteration
    * - :ini:`backend = cashocs`
      - specifies the backend for solving nonlinear equations, can be either :ini:`cashocs` or :ini:`petsc`

[OptimizationRoutine]
*********************

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`algorithm`
      - has to be specified by the user; see :py:meth:`solve <cashocs.OptimalControlProblem.solve>`
    * - :ini:`rtol = 1e-3`
      - relative tolerance for the optimization algorithm
    * - :ini:`atol = 0.0`
      - absolute tolerance for the optimization algorithm
    * - :ini:`max_iter = 100`
      - maximum iterations for the optimization algorithm
    * - :ini:`gradient_method = direct`
      - specifies the solver for computing the gradient, can be either :ini:`gradient_method = direct` or :ini:`gradient_method = iterative`
    * - :ini:`gradient_tol = 1e-9`
      - the relative tolerance in case an iterative solver is used to compute the gradient.
    * - :ini:`soft_exit = False`
      - if :ini:`soft_exit = True`, the optimization algorithm does not raise an exception if
        it did not converge

        
[LineSearch]
************

.. list-table::
    :header-rows: 1
    
    * - Parameter = Default value
      - Remarks
    * - :ini:`method = armio`
      - :ini:`method = armijo` is a simple backtracking line search, whereas :ini:`method = polynomial` uses polynomial models to compute trial stepsizes.
    * - :ini:`initial_stepsize = 1.0`
      - initial stepsize for the first iteration in the Armijo rule
    * - :ini:`epsilon_armijo = 1e-4`
      -
    * - :ini:`beta_armijo = 2.0`
      -
    * - :ini:`safeguard_stepsize = True`
      - De(-activates) a safeguard against poor scaling
    * - :ini:`polynomial_model = cubic`
      - This specifies, whether a cubic or quadratic model is used for computing trial stepsizes
    * - :ini:`factor_high = 0.5`
      - Safeguard for stepsize, upper bound
    * - :ini:`factor_low = 0.1`
      - Safeguard for stepsize, lower bound
    * - :ini:`fail_if_not_converged = False`
      - if this is :ini:`True`, then the line search fails if the state system can not be solved at the new iterate


[AlgoLBFGS]
***********

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`bfgs_memory_size = 5`
      - memory size of the L-BFGS method
    * - :ini:`use_bfgs_scaling = True`
      - if :ini:`use_bfgs_scaling = True`, uses a scaled identity mapping as initial guess for the inverse Hessian
    * - :ini:`bfgs_periodic_restart = 0`
      - specifies, after how many iterations the method is restarted. If this is 0, no restarting is done.
    * - :ini:`damped = False`
      - specifies whether damping for the BFGS method should be used to enforce the curvature condition and prevent restarting


[AlgoCG]
********

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`cg_method = FR`
      - specifies which nonlinear CG method is used
    * - :ini:`cg_periodic_restart = False`
      - if :ini:`cg_periodic_restart = True`, enables periodic restart of NCG method
    * - :ini:`cg_periodic_its = 10`
      - specifies, after how many iterations the NCG method is restarted, if applicable
    * - :ini:`cg_relative_restart = False`
      - if :ini:`cg_relative_restart = True`, enables restart of NCG method based on a relative criterion
    * - :ini:`cg_restart_tol = 0.25`
      - the tolerance of the relative restart criterion, if applicable

[AlgoTNM]
*********

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`inner_newton = cr`
      - inner iterative solver for the truncated Newton method
    * - :ini:`inner_newton_rtol = 1e-15`
      - relative tolerance for the inner iterative solver
    * - :ini:`inner_newton_atol = 0.0`
      - absolute tolerance for the inner iterative solver
    * - :ini:`max_it_inner_newton = 50`
      - maximum iterations for the inner iterative solver

[Output]
********

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`verbose = True`
      - if :ini:`verbose = True`, the history of the optimization is printed to the console
    * - :ini:`save_results = True`
      - if :ini:`save_results = True`, the history of the optimization is saved to a .json file
    * - :ini:`save_txt = True`
      - if :ini:`save_txt = True`, the history of the optimization is saved to a human readable .txt file
    * - :ini:`save_state = False`
      - if :ini:`save_state = True`, the history of the state variables over the optimization is
        saved in .xdmf files
    * - :ini:`save_adjoint = False`
      - if :ini:`save_adjoint = True`, the history of the adjoint variables over the optimization is
        saved in .xdmf files
    * - :ini:`save_gradient = False`
      - if :ini:`save_gradient = True`, the history of the gradient(s) over the optimization is saved in .xdmf files
    * - :ini:`result_dir = ./`
      - path to the directory, where the output should be placed
    * - :ini:`precision = 3`
      - number of significant digits to be printed
    * - :ini:`time_suffix = False`
      - Boolean flag, which adds a suffix to :ini:`result_dir` based on the current time


This concludes the documentation of the config files for optimal control problems.
For the corresponding documentation for shape optimization problems, see :ref:`config_shape_optimization`.
