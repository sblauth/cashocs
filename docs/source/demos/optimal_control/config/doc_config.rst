.. _config_optimal_control:

Documentation of the Config Files for Optimal Control Problems
==============================================================


Let us take a look at how the config files are structured for optimal control
problems.
The corresponding config file is :download:`config.ini
</../../demos/documented/optimal_control/poisson/config.ini>`.

First of all, the config is divided into three sections: :ref:`Mesh
<config_ocp_mesh>`, :ref:`StateEquation <config_ocp_state_equation>`,
and :ref:`OptimizationRoutine <config_ocp_optimization_routine>`.
These manage the settings for the mesh, the state equation of the optimization
problem, and for the solution algorithm, respectively.




.. _config_ocp_mesh:

Section Mesh
------------
The mesh section consists, for optimal control problems, only of a path to the
.xdmf version of the mesh file ::

    [Mesh]
    mesh_file = ../mesh/mesh.xdmf

This section is completely optional and can be used when importing GMSH
generated meshes. Note, that this section can become more populated and useful
for shape optimization problems, as detailed in the
:ref:`description of their config files <config_shape_mesh>`. To convert a .msh
file to the .xdmf format, you can use the built-in converter as ::

    cashocs-convert gmsh_file.msh xdmf_file.xdmf

from the command line.



.. _config_ocp_state_equation:

Section StateEquation
---------------------
The state equation section is used to detail how the state systems and, hence, also the
adjoint systems are solved. This includes settings for a Picard iteration. The section
starts as usual with the following command ::

    [StateEquation]

In the following, we go over each parameter in detail. First, we have ::

    is_linear = True

This is a boolean parameter which indicates, whether the state system
is linear. This is used to speed up some computations. Note, that the program
will always work when this is set to False, as it treats the linear problem in a
nonlinear fashion and converges in one iteration. However, using
``is_linear = True``
on a nonlinear state system throws a fenics error.

The next parameter is defined via ::

    inner_newton_atol = 1e-13

This parameter determines the absolute tolerance for the Newton solver that is
used to solve a nonlinear state system.

Subsequently, we can also defined the relative tolerance for the Newton solver
via ::

    inner_newton_rtol = 1e-11

Moreover, we have the following parameters for the Newton method ::

    newton_damped = True

which determines if a damping should be used (in case this is ``True``) or not
(otherwise). This defaults to ``True`` if nothing is given. The parameter ::

    newton_verbose = False

is used to make the Newton solver's output verbose. This is disabled by default.

Finally, the parameter ::

    newton_iter = 50

controls how many iterations the Newton method is allowed to make before it
terminates. This defaults to 50.


Next, we have ::

    picard_iteration = False

This is another boolean flag. This is used to determine, whether the state system
shall be solved using a Picard iteration (if this is ``True``) or not
(if this is ``False``). For a single state equation (i.e. one single state
variable) both options are equivalent. The difference is only active when
considering a coupled system with multiple state variables that is coupled.

The tolerances for the Picard iteration are defined via ::

    picard_rtol = 1e-10
    picard_atol = 1e-12

The first parameter determines the relative tolerance used for the Picard
iteration, in case it is enabled, and the second one determines the absolute
tolerance.


The maximum number of iterations of the method can be set via ::

    picard_iter = 10

The parmater ``picard_verbose`` enables verbose output of the convergence of the
Picard iteration, and is set as follows ::

    picard_verbose = False




.. _config_ocp_optimization_routine:

Section OptimizationRoutine
---------------------------

The final section is the heart of the solution algorithm, which can be
customized here. It starts with ::

    [OptimizationRoutine]

The first parameter determines the choice of the particular algorithm, via ::

    algorithm = lbfgs

The possible choices are given by

  - ``gd`` or ``gradient_descent`` : a gradient descent method

  - ``cg``, ``conjugate_gradient``, ``ncg``, ``nonlinear_cg`` : nonlinear CG methods

  - ``lbfgs`` or ``bfgs`` : limited memory BFGS method

  - ``newton`` : a truncated Newton method

  - ``pdas`` or ``primal_dual_active_set`` : a primal dual active set method (for control constraints)

Next up, we have ::

    maximum_iterations = 250

This parameter determines the maximum number of iterations carried out by the
solution algorithm before it is terminated.

The next line of the config file is given by ::

    rtol = 1e-4

This parameter determines the relative tolerance for the solution algorithm.
In the case where no control constraints are present, this uses the "classical"
norm of the gradient of the cost functional as measure. In case there are box
constraints present, it uses the stationarity measure (see `Kelley, Iterative Methods
for Optimization <https://doi.org/10.1137/1.9781611970920>`_ as measure.

Analogously, we also have the line ::

    atol = 0.0

This determines the absolute tolerance for the solution algorithm.

The initial step size can be set via ::

    step_initial = 1.0

This parameter determines the initial step size to be used in the line search.
This can have an important effect on performance of the gradient descent and nonlinear
cg methods, as they do not include a built-in scaling of the step size.

The next paramter, ``epsilon_armijo``, is defined as follows ::

    epsilon_armijo = 1e-4

This paramter describes the parameter used in the Armijo rule to determine
sufficient decrease, via

.. math:: J(u + td) \leq J(u) + \varepsilon t \left\langle g, d \right\rangle

where u is the current optimization variable, d is the search direction, t is the
step size, and g is the current gradient. Note, that :math:`\varepsilon` is the parameter
``epsilon_armijo``.
A value of 1e-4 is recommended and commonly used (see `Nocedal and Wright,
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).

In the following line, the parameter ``beta_armijo`` is defined ::

    beta_armijo = 2

This parameter determines the factor by the which the step size is decreased
if the Armijo condition is not satisfied, i.e., we get ``t = t / beta`` as new
step size.

Next up, we have the parameter ``soft_exit``, which is defined as ::

    soft_exit = True

This parameter determines, whether we get a hard (``False``) or soft (``True``) exit
of the optimization routine in case it does not converge. In case of a hard exit
an Exception is raised and the script does not complete. However, it can be beneficial
to still have the subsequent code be processed, which happens in case ``soft_exit = True``.
Note, however, that in this case the returned results are **NOT** optimal,
as defined by the user input parameters.

The next line reads ::

    verbose = True

The parameter `verbose` determines, whether the solution algorithm generates a verbose
output in the console, useful for monitoring its convergence.

Next up, we define the paramter ``save_results`` ::

    save_results = False

If this parameter is set to True, the history of the optimization is saved in
a .json file located in the same folder as the optimization script. This is
very useful for postprocessing the results.

Afterwards, we define the parameter ``save_pvd`` in the line ::

    save_pvd = False

If ``save_pvd`` is set to True, the state variables are saved to .pvd files
in a folder named "pvd", located in the same directory as the optimization script.


The following sections describe parameters that belong to the certain solution
algorithms, and are also specified under the OptimizationRoutine section.

Limited memory BFGS method
**************************


For the L-BFGS method we have the following parameters. First, we have
``memory_vectors``, which is set via ::

    memory_vectors = 2

and determines the size of the memory of the L-BFGS method. E.g., the command
above specifies that information of the previous two iterations shall be used.
The case ``memory_vectors = 0`` yields the classical gradient descent method,
whereas memory_vectors > maximum_iterations gives rise to the classical
BFGS method with unlimited memory.

Second, we have the parameter ``use_bfgs_scaling``, that is set via ::

    use_bfgs_scaling = True

This determines, whether one should use a scaling of the initial Hessian approximation
(see `Nocedal and Wright, Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is usually very beneficial and should be kept enabled.

Nonlinear conjugate gradient methods
************************************

The parameter ::

    cg_method = PR

determines which of the nonlinear cg methods shall be used. Available are

- ``FR`` : the Fletcher-Reeves method

- ``PR`` : the Polak-Ribiere method

- ``HS`` : the Hestenes-Stiefel method

- ``DY`` : the Dai-Yuan method

- ``HZ`` : the Hager-Zhang method


After the definition of the particular cg method, we now have parameters determining
restart strategies for these method. First up, we have the line ::

    cg_periodic_restart = False

This parameter determines, whether the CG method should be restarted with a gradient
step periodically, which can lead to faster convergence. The amount of iterations
between restarts is then determined by ::

    cg_periodic_its = 5

In this example, the NCG method is restarted after 5 iterations.

Another possibility to restart NCG methods is based on a relative criterion
(see `Nocedal and Wright,
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is enabled via the boolean flag ::

    cg_relative_restart = False

and the corresponding relative tolerance (which should lie in :math:`(0,1)`) is determined via ::

    cg_restart_tol = 0.5

Note, that this relative restart reinitializes the iteration with a gradient
step in case subsequent gradients are not "sufficiently" orthogonal anymore.


Truncated Newton method
***********************

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
initial guess is further away from the optimum.

In the next line, the paramter ``max_it_inner_newton`` is defined via ::

    max_it_inner_newton = 50

This parameter determines how many iterations of the Krylov solver are performed
before the inner iteration is terminated. Note, that the approximate solution
of the Hessian problem is used after ``max_it_inner_newton`` iterations regardless
of whether this is converged or not.

Finally, we have the following line ::

    inner_newton_tolerance = 1e-15

This determines the relative tolerance of the iterative Krylov solver for the
Hessian problem.

Primal-Dual-Active-Set Method
*****************************


Finally, we take a look at the parameters for the primal dual active set method.
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

    algorithm = pdas
    inner_pdas = bfgs
    memory_vectors = 2

uses the limited memory BFGS method with memory size 2 as inner solver for the
primal dual active set method.

The maximum number of (inner) iterations for the primal dual active set method are
defined via ::

    maximum_iterations_inner_pdas = 100

Next up, we have the following line ::

    pdas_shift_mult = 1e-4

This determines the shift multiplier for the determination of the active and
inactive sets, usually denoted by :math:`\gamma`, and should be positive. This comes from
the interpretation as semi-smooth Newton method with Moreau Yosida regularization
of the constraints.

Finally, we have the parameter ::

    pdas_inner_tolerance = 1e-2

This parameter determines the relative tolerance used for the inner
solution algorithms.

This concludes the documentation of the config files for optimal control problems.
For the corresponding documentation for shape optimization problems, see :ref:`config_shape_optimization`.