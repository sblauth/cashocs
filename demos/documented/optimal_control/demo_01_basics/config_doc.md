Documentation of the config files for optimal control problems
==============================================================

Let us take a look at how the config files are structured for optimal control problems. 
The corresponding config file is located at './config.ini'.

First of all, the config is divided into three sections: Mesh, StateEquation, and OptimizationRoutine.
These manage the settings for the mesh, the state equation of the optimization problem, and for the solution
algorithm, respectively.

Mesh
----

The mesh section consists, for optimal control problems, only of a path to the .xdmf version of the mesh file

    [Mesh]
    mesh_file = ../mesh/mesh.xdmf
    
This section is completely optional and can be used when importing GMSH generated meshes. 
To convert a .msh file to the .xdmf format, you can use the built-in converter as

    mesh-convert gmsh_file.msh xdmf_file.xdmf
   
from the command line.


StateEquation
-------------

The state equation section is used to detail how the state systems and, hence, also the
adjoint systems are solved. This includes settings for a Picard iteration. The section 
starts as usual with the following command

    [StateEquation]

In the following, we go over each parameter in detail. First, we have

    is_linear = true

This is a boolean parameter which indicates, whether the state equation / system
is linear. This is used to speed up some computations. Note, that the program will 
always work when this is set to false, as it treats the linear problem in a nonlinear
fashion and converges in one iteration. However, using is_linear = true on a nonlinear
state system throws a fenics error.

    inner_newton_atol = 1e-13

This parameter determines the absolute tolerance for the Newton solver that is used
to solve a nonlinear state system.

    inner_newton_rtol = 1e-11

This parameter determines the relative tolerance for the Newton solver used to
solve a nonlinear state system.

Next, we have

    picard_iteration = false
   
This is another boolean flag. This is used to determine, whether the state system
shall be solved using a Picard iteration (true) or not (false). For a single 
state equation (i.e. one single state variable) both options are equivalent. 
Moreover, for an uncoupled system picard_iteration = true converges in a single
Picard iteration so that it is also equivalent to picard_iteration = false. The difference
is only active when considering a coupled system with multiple state variables
that is coupled.

    picard_rtol = 1e-10

This parameter determines the relative tolerance used for the Picard iteration, in case
this is enabled. 

    picard_atol = 1e-12

This parameter determines the absolute tolerance used for the Picard iteration, in case
this is enabled.

    picard_iter = 10

This paramter determines the maximum amount of iterations that are carried out to
solve the state system before the iteration is terminated.

    picard_verbose = false

This enables verbose output of the convergence of the Picard iteration

Note, that it is currently not possible to determine custom krylov solvers 
for the state system, but this is planned in a future release, and will be
available via the config files.


OptimizationRoutine
-------------------

The final section is the heart of the solution algorithm, which can be customized here.
It starts with

    [OptimizationRoutine]

The first parameter determines the choice of the particular algorithm, via

    algorithm = lbfgs

The possible choices are given by 
- 'gd' or 'gradient_descent' : A gradient descent method

- 'cg' or 'conjugate_gradient' : Nonlinear CG methods

- 'lbfgs' or 'bfgs' : limited memory BFGS method

- 'newton' : a truncated Newton method

- 'semi_smooth_newton' : a semi-smooth newton method

- 'pdas' or 'primal_dual_active_set' : a primal dual active set method
    (for control constraints)


    maximum_iterations = 250

This parameter determines the maximum number of iterations carried out by the 
solution algorithm before it is terminated.

    rtol = 1e-4

This parameter determines the relative tolerance for the solution algorithm.
In the case where no control constraints are present, this uses the "classical"
norm of the gradient of the cost functional as measure. In case there are box 
constraints present, it uses the stationarity measure (see Kelley, Iterative Methods 
for Optimization, Chapter 4) as measure.

    atol = 0.0

This determines the absolute tolerance for the solution algorithm. The corresponding
measures are chosen analogously to the relative tolerance above.

    step_initial = 1.0

This parameter determines the initial step size to be used in the line search.
This can have an important effect on performance of "first order" algorithms

    epsilon_armijo = 1e-4

This paramter describes the parameter used in the Armijo rule to determine
sufficient decrease, via
_J(u + td) <= J(u) + eps * t * <g, d>,_
where u is the current optimization variable, d is the search direction, t is the
step size, and g is the current gradient. eps is the parameter determined above.
A value of 1e-4 is recommended and commonly used (see Nocedal and Wright, 
Numerical Optimization).

    beta_armijo = 2

This parameter determines the factor by the which the step size is decreased 
if the Armijo condition is not satisfied, i.e., we get _t = t / beta_ as new
step size.

    soft_exit = true

This parameter determines, whether we get a hard (false) or soft (true) exit
of the optimization routine in case it does not converge. In case of a hard exit
a SystemExit is raised and the script does not complete. However, it can be beneficial
to still have the subsequent code be processed, which happens in case soft_exit = true.
Note, however, that in this case the returned results are **NOT** optimal, 
as defined by the user input parameters.

    verbose = true

This parameter determines, whether the solution algorithm generates a verbose
output in the console, useful for monitoring its convergence.

    save_results = false

If this parameter is set to true, the history of the optimization is saved in
a .json file located in the same folder as the optimization script. This is
very useful for postprocessing the results.

    save_pvd = false

If this is set to true, the state variables are saved to .pvd files
in the folder "pvd", located in the same directory as the optimization script.


The following sections describe parameters that belong to the certain solution
algorithms, and are also specified under the [OptimizationRoutine] section.

Limited memory BFGS method
--------------------------

For the L-BFGS method we have the following parameters.

    memory_vectors = 2

Determines the size of the memory of the L-BFGS method. E.g., the command 
above specifies that information of the previous two iterations shall be used.
The case memory_vectors = 0 yields the classical gradient descent method,
whereas memory_vectors > maximum_iterations gives rise to the classical
BFGS method with unlimited memory. 

    use_bfgs_scaling = true

This determines, whether one should use a scaling of the initial Hessian approximation
(see Nocedal and Wright, Numerical Optimization). This is usually very beneficial
and should be kept enabled.

Nonlinear conjugate gradient methods
------------------------------------

    cg_method = PR

Determines which of the nonlinear cg methods shall be used. Available are

- 'FR' : The Fletcher-Reeves method

- 'PR' : The Polak-Ribiere method

- 'HS' : The Hestenes-Stiefel method

- 'DY' : The Dai-Yuan method

- 'HZ' : The Hager-Zhang method


    cg_periodic_restart = False

This parameter determines, whether the CG method should be restarted with a gradient
step periodically, which can lead to faster convergence. The amount of iterations
between restarts is then determined by 

    cg_periodic_its = 5

In this example, the NCG method is restarted after 5 iterations.

Another possibility to restart NCG methods is based on a relative criterion
(see Nocedal and Wright, Numerical Optimization). This is enabled via the flag

    cg_relative_restart = False

and the corresponding relative tolerance (which should lie in (0,1)) is determined via

    cg_restart_tol = 0.5

Note, that this relative restart reinitializes the iteration with a gradient 
step in case subsequent gradients are not "sufficiently" orthogonal anymore.


Truncated Newton method
-----------------------

The parameters for both the classical truncated Newton method and for the semi-smooth
Newton method are determined in the following.

    inner_newton = cg

Determines the Krylov method for the solution of the Newton problem. Should be one
of 

- 'cg' : A linear conjugate gradient method

- 'minres' : A minimal residual method

- 'cr' : A conjugate residual method

Note, that all of these Krylov solvers are streamlined for symmetric linear
operators, which the Hessian is (should be also positive definite for a minimizer
so that the conjugate gradient method should yield good results when initialized
not too far from the optimum)

    max_it_inner_newton = 50

This parameter determines how many iterations of the Krylov solver are performed
before the inner iteration is terminated. Note, that the approximate solution
of the Hessian problem is used after max_it_inner_newton iterations regardless
of whether this is converged or not

    inner_newton_tolerance = 1e-15

This determines the relative tolerance of the iterative Krylov solver for the
Hessian problem.

Primal-Dual-Active-Set Method
-----------------------------

Finally, we take a look at the parameters for the primal dual active set method.

    inner_pdas = newton

This parameter determines which solution algorithm is used for the inner 
(unconstrained) optimization problem in the primal dual active set method.
Can be one of

- 'gd' or 'gradient_descent' : A gradient descent method

- 'cg' or 'conjugate_gradient' : A nonlinear conjugate gradient method

- 'lbfgs' or 'bfgs' : A limited memory BFGS method

- 'newton' : A truncated newton method

Note, that the parameters for these inner solvers are determined via the same
interfaces used for the solution algorithms, i.e, setting

    algorithm = pdas
    inner_pdas = bfgs
    memory_vectors = 2

uses the limited memory BFGS method with memory size 2 as inner solver for the 
primal dual active set method.

    maximum_iterations_inner_pdas = 100

This parameter detemines the maximum amount of iterations performed by the
inner solution algorithm for the sub-problems encountered in the primal
dual active set method. 

    pdas_shift_mult = 1e-4

This determines the shift multiplier for the determination of the active and
inactive sets, usually denoted by \gamma, and should be positive. This comes from
the interpretation as semi-smooth Newton method with Moreau Yosida regularization
of the constraints.

    pdas_inner_tolerance = 1e-2

This parameter determines the relative tolerance used for the inner
solution algorithms. 
