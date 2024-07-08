.. _config_shape_optimization:

Documentation of the Config Files for Shape Optimization Problems
=================================================================

Let us take a detailed look at the config files for shape optimization problems and
discusss the corresponding parameters.

For shape optimization problems, the config file is a lot larger compared to the :ref:`config files
for optimal control <config_optimal_control>`.
However, several important parameters are shared between both _typing of optimization
problems.

A general config file for shape optimization has the following sections
:ref:`Mesh <config_shape_mesh>`, :ref:`StateSystem <config_shape_state_equation>`,
:ref:`OptimizationRoutine <config_shape_optimization_routine>`, :ref:`AlgoLBFGS <config_shape_algolbfgs>`,
:ref:`AlgoCG <config_shape_algocg>`,
:ref:`ShapeGradient <config_shape_shape_gradient>`,
:ref:`Regularization <config_shape_regularization>`, :ref:`MeshQuality <config_shape_mesh_quality>`
and :ref:`Output <config_shape_output>`. We go over these
sections and each parameter in them in the following.

As in :ref:`config_optimal_control`, we refer to the `documentation of the
configparser module <https://docs.python.org/3/library/configparser.html>`_ for
a detailed description of how these config files can be structured. Moreover,
we remark that cashocs has a default behavior for almost all of these
parameters, which is triggered when they are **NOT** specified in the config file,
and we will discuss this behavior for each parameter in this tutorial. For a
summary over all parameters and their default values look at
:ref:`the end of this page <config_shape_summary>`.



.. _config_shape_mesh:

Section Mesh
------------

The mesh section is, in contrast to the corresponding one for optimal control problems,
more important and also has more parameters. However, its primary importance is for
remeshing, which we cover in :ref:`demo_remeshing`.

As first parameter, we have

.. code-block:: ini

    mesh_file = ./mesh/mesh.xdmf

This specifies a path to a .xdmf file containing the discretized geometry. For all purposes, cashocs assumes that this .xdmf file was generated via conversion from a
GMSH file using the command line command :py:func:`cashocs.convert`.

Note, that the corresponding files for the boundaries and subdomains are generated
automatically with :py:func:`cashocs.convert`, and they will also be read by :py:func:`import_mesh <cashocs.import_mesh>`
if they are present.


The second parameter in the Mesh section, :ini:`gmsh_file`, is defined via

.. code-block:: ini

    gmsh_file = ./mesh/mesh.msh

This defines the path to the GMSH .msh file which was used to create the .xdmf file
specified in :ini:`mesh_file`. As before, this parameter is only relevant for remeshing
purposes, and not needed otherwise.

The next parameter is :ini:`geo_file`, which is the final file we need for remeshing (
and only there). It is also given by a path to a file, in this case to the GMSH .geo
file used to generate the :ini:`gmsh_file`. It is specified, .e.g., as

.. code-block:: ini

    geo_file = ./mesh/mesh.geo

.. note::

    For a detailed discussion of how to use these parameters we refer to :ref:`demo_remeshing`.

Next up is a boolean flag that is used to indicate whether remeshing shall be performed

.. code-block:: ini

    remesh = False


As the remeshing feature is experimental, we do advise to always try without
remeshing. Note, that by default this flag is set to :ini:`remehs = False` so that remeshing is disabled.

Finally, we have the boolean flag :ini:`show_gmsh_output`, specified via

.. code-block:: ini

    show_gmsh_output = False

This is used to toggle on / off the terminal output of GMSH when it performs a
remeshing operation. This can be helpful for debugging purposes. By default, this
is set to :ini:`show_gmsh_output = False`.

As stated throughout the Mesh section, these parameters are optional most of the time,
and are only really required for remeshing. You can safely leave them out of your config file, and you should not need them, unless you want to perform remeshing.


.. _config_shape_state_equation:

Section StateSystem
---------------------

The StateSystem section is in complete analogy to :ref:`the corresponding one for optimal control problems <config_ocp_state_system>`. For the
sake of completeness, we briefly recall the parameters here, anyway.

The first parameter is :ini:`is_linear`, and can be set as

.. code-block:: ini

    is_linear = True

This is a boolean flag that indicates whether the state system is linear or not.
The default value for this parameter is :ini:`is_linear = False`, as every linear problem can also be
interpreted as a nonlinear one.

The next parameters are used to define the tolerances of the Newton solver, in
case a nonlinear state system has to be solved

.. code-block:: ini

    newton_rtol = 1e-11
    newton_atol = 1e-13


Here, :ini:`newton_rtol` sets the relative, and :ini:`newton_atol` the absolute tolerance
for Newton's method. Their default values are :ini:`newton_rtol = 1e-11` and
:ini:`newton_atol = 1e-13`.

The next parameter for the Newton iteration is the maximum number of iterations it
is allowed to perform before the iteration is cancelled. This is controlled via

.. code-block:: ini

    newton_iter = 50

which defaults to :ini:`newton_iter = 50`.

The parameter :ini:`newton_damped`, which is set via

.. code-block:: ini

    newton_damped = True

is a boolean flag, indicating whether a damping strategy should be performed for the
Newton method, or whether the classical Newton-Raphson iteration shall be used. This
defaults to :ini:`newton_damped = False` (as this is faster), but for some problems it might be beneficial to
use damping in order to enhance the convergence of the nonlinear solver.

Additionally, we have the boolean parameter :ini:`newton_inexact`, defined via

.. code-block:: ini

    newton_inexact = False

which sets up an inexact Newton method for solving nonlinear problems in case this is :ini:`newton_inexact = True`. The default is :ini:`newton_inexact = False`.

Next, we have the parameter

.. code-block:: ini

    newton_verbose = False

This is used to toggle the verbose output of the Newton method for the state system.
By default this is set to :ini:`newton_verbose = False` so that there is not too much noise in the terminal.


The upcoming parameters are used to define the behavior of a Picard iteration, that
may be used if we have multiple variables.

.. note::

    For a detailed discussion of how to use the Picard iteration to solve a coupled
    state system, we refer to :ref:`demo_picard_iteration`. Note, that this demo
    is written for optimal control problems, but the definition of the state system
    can be transferred analogously to shape optimization problems, too.

First, we have a boolean flag, set via

.. code-block:: ini

    picard_iteration = False

which determines whether the Picard iteration is enabled or not. This defaults
to :ini:`picard_iteration = False`, so that the Picard solver is disabled by default.
The following two parameters determine, analogously to above, the tolerances for the
Picard iteration

.. code-block:: ini

    picard_rtol = 1e-10
    picard_atol = 1e-12

The default values for these parameters are :ini:`picard_rtol = 1e-10` and
:ini:`picard_atol = 1e-12`. Moreover, note that the tolerances of the Newton solver are adjusted automatically in case
a Picard iteration is performedm, so that an inexact Picard iteration is used.

The maximum amout of iterations for the Picard iteration are set with

.. code-block:: ini

    picard_iter = 10

The default value for this is given by :ini:`picard_iter = 50`.

Finally, we can enable verbose output of the Picard iteration with the following
boolean flag

.. code-block:: ini

    picard_verbose = False

which is set to :ini:`picard_verbose = False` by default.

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


.. _config_shape_optimization_routine:

Section OptimizationRoutine
---------------------------

The section OptimizationRoutine also closely resembles :ref:`the one for optimal control
problems <config_ocp_optimization_routine>`. Again, we will take a brief look at all parameters here

The first parameter that can be controlled via the config file is :ini:`algorithm`, which is
set via

.. code-block:: ini

    algorithm = lbfgs

There are three possible choices for this parameter for shape optimization problems, namely

- :ini:`algorithm = gd` or :ini:`algorithm = gradient_descent` : A gradient descent method

- :ini:`algorithm = cg`, :ini:`algorithm = conjugate_gradient`, :ini:`algorithm = ncg`, :ini:`algorithm = nonlinear_cg` : Nonlinear CG methods

- :ini:`algorithm = lbfgs` or :ini:`algorithm = bfgs` : limited memory BFGS method.


Thereafter, we specify the tolerances for the optimization algorithm with the parameters

.. code-block:: ini

    rtol = 5e-3
    atol = 0.0

Again, :ini:`rtol` denotes the relative, and :ini:`atol` the absolute tolerance, and the
defaults for these parameters are given by :ini:`rtol = 1e-3`, and :ini:`atol = 0.0`.

The next parameter is used to control the maximum number of iterations performed by
the optimization algorithm. It is set via

.. code-block:: ini

    max_iter = 50

and defaults to :ini:`max_iter = 100`.

Next up, we have the initial guess for the step size, which can be determined via

.. code-block:: ini

    initial_stepsize = 1.0

The default behavior is given by :ini:`initial_stepsize = 1.0`.

The next parameter is given by

.. code-block:: ini

    safeguard_stepsize = True
    
This parameter can be used to activate safeguarding of the initial stepsize for line search methods. This helps
to choose an apropriate stepsize for the initial iteration even if the problem is poorly scaled. 

The upcoming parameters are used for the Armijo rule

.. code-block:: ini

    epsilon_armijo = 1e-4
    beta_armijo = 2

They are used to verify that the condition

.. math:: J((I + t \mathcal{V})\Omega) \leq J(\Omega) + \varepsilon_{\text{Armijo}}\ t\ dJ(\Omega)[\mathcal{V}]

holds, and if this is not satisfied, the stepsize is updated via :math:`t = \frac{t}{\beta_{\text{Armijo}}}`.
As default values for these parameters we use :ini:`epsilon_armijo = 1e-4` as well
as :ini:`beta_armijo = 2`.

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

The following parameter, :ini:`soft_exit`, is a boolean flag which determines how
the optimization algorithm is terminated in case it does not converge. If :ini:`soft_exit = True`, then an
error message is printed, but code after the :py:meth:`solve <cashocs.ShapeOptimizationProblem.solve>` call of the
optimization problem will still be executed. However, when :ini:`soft_exit = False`, cashocs
raises an exception and terminates. This is set via 

.. code-block:: ini

    soft_exit = False

and is set to :ini:`soft_exit = False` by default.


.. _config_sop_linesearch:

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

.. _config_shape_algolbfgs:

Section AlgoLBFGS
-----------------

Next, we discuss the parameters relevant for the limited memory BFGS method. For details
regarding this method, we refer to `Schulz, Siebenborn, and Welker, Efficient PDE Constrained Shape Optimization Based on Steklov-Poincaré-Type Metrics
<https://doi.org/10.1137/15M1029369>`_, where the methods are introduced.

The first parameter, :ini:`bfgs_memory_size`, determines how large the storage of the BFGS method is. It is set via

.. code-block:: ini

    bfgs_memory_size = 3

Usually, a higher storage leads to a better Hessian approximation, and thus to faster
convergence. However, this also leads to an increased memory usage. Typically, values
below 5 already work very well. The default is :ini:`bfgs_memory_size = 5`.

The other parameter for the BFGS method is

.. code-block:: ini

    use_bfgs_scaling = True

This determines, whether one should use a scaling of the initial Hessian approximation
(see `Nocedal and Wright, Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is usually very beneficial and should be kept enabled (which is the default).

Third, we have the parameter :ini:`bfgs_periodic_restart`, which is set in the line

.. code-block:: ini

    bfgs_periodic_restart = 0
   
This is a non-negative integer value, which indicates the number of BFGS iterations, before a reinitialization takes place. In case that this is :ini:`bfgs_periodic_restart = 0` (which is the default), no restarts are performed. 

Finally, we have the parameter :ini:`damped`, which can be set with

.. code-block:: ini

    damped = False

This parameter is a boolean flag, which indicates whether Powell's damping (on H) should be used or not. This is useful, when the curvature condition is not satisfied and (without damping) a restart would be required. The default is :ini:`damped = False`.

.. _config_shape_algocg:

Section AlgoCG
--------------

The following parameters are used to define the behavior of the nonlinear conjugate
gradient methods for shape optimization. For more details on this, we refer to the
preprint `Blauth, Nonlinear Conjugate Gradient Methods for PDE Constrained Shape
Optimization Based on Steklov-Poincaré-Type Metrics <https://arxiv.org/abs/2007.12891>`_.

First, we define which nonlinear CG method is used by

.. code-block:: ini

    cg_method = DY

Available options are

- :ini:`cg_method = FR` : The Fletcher-Reeves method

- :ini:`cg_method = PR` : The Polak-Ribiere method

- :ini:`cg_method = HS` : The Hestenes-Stiefel method

- :ini:`cg_method = DY` : The Dai-Yuan method

- :ini:`cg_method = HZ` : The Hager-Zhang method

The default value is :ini:`cg_method = FR`. As for optimal control problems, the subsequent parameters are used to define the
restart behavior of the nonlinear CG methods. First, we have

.. code-block:: ini

    cg_periodic_restart = False

This boolean flag en- or disables that the NCG methods are restarted after a fixed
amount of iterations, which is specified via

.. code-block:: ini

    cg_periodic_its = 5

i.e., if :ini:`cg_periodic_restart = True` and :ini:`cg_periodic_its = n`, then the NCG method
is restarted every :math:`n` iterations. The default behavior is given by
:ini:`cg_periodic_restart = False` and :ini:`cg_periodic_its = 10`.

Alternatively, there also exists a relative restart criterion (see `Nocedal and Wright,
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_), which can be enabled
via the boolean flag :ini:`cg_relative_restart`, which is defined in the line

.. code-block:: ini

    cg_relative_restart = False

and the corresponding restart tolerance is set in

.. code-block:: ini

    cg_restart_tol = 0.5

Note, that :ini:`cg_restart_tol` should be in :math:`(0, 1)`. If two subsequent
gradients generated by the nonlinear CG method are not "sufficiently
orthogonal", the method is restarted with a gradient step. The default behavior
is given by :ini:`cg_relative_restart = False` and :ini:`cg_restart_tol = 0.25`.

.. _config_shape_shape_gradient:

Section ShapeGradient
---------------------

After we have specified the behavior of the solution algorithm, this section
is used to specify parameters relevant to the computation of the shape gradient.
Note, that by shape gradient we refer to the following object.

Let :math:`\mathcal{S} \subset \{ \Omega \;\vert\; \Omega \subset \mathbb{R}^d \}` be a
subset of the power set of :math:`\mathbb{R}^d`. Let :math:`J` be a shape differentiable functional
:math:`J \colon \mathcal{S} \to \mathbb{R}` with shape derivative :math:`dJ(\Omega)[\mathcal{V}]`.
Moreover, let :math:`a \colon H \times H \to \mathbb{R}` be a symmetric, continuous, and
coercive bilinear form on the Hilbert space :math:`H`.
Then, the shape gradient :math:`\mathcal{G}` of :math:`J` (w.r.t. :math:`a`) is defined as the solution of the
problem

.. math::

    \text{Find } \mathcal{G} \in H \text{ such that } \\
    \quad a(\mathcal{G}, \mathcal{V}) = dJ(\Omega)[\mathcal{V}].


For PDE constrained shape optimization, it is common to use a bilinear form based on
the linear elasticity equations, which enables smooth mesh deformations. This bilinear
form is given as follows, in a general form, that is also implemented in cashocs

.. math::

    a \colon H \times H; \quad a(\mathcal{W}, \mathcal{V}) = \int_\Omega
    2 \mu \left( \varepsilon(\mathcal{W}) : \varepsilon(\mathcal{V}) \right) + \lambda \left( \text{div}(\mathcal{W}) \text{div}(\mathcal{V}) \right) + \delta \left( V \cdot W \right) \text{ d}x,

where :math:`H` is some suitable subspace of :math:`H^1(\Omega)^d` and :math:`\varepsilon(\mathcal{V}) = \frac{1}{2}(D\mathcal{V} + D\mathcal{V}^\top)`
is the symmetric part of the Jacobian.
The subspace property is needed
to include certain geometrical constraints of the shape optimization problem, which fix
certain boundaries, into the shape gradient. For a detailed description of this
setting we refer to the preprint `Blauth, Nonlinear Conjugate Gradient Methods for PDE
Constrained Shape Optimization Based on Steklov-Poincaré-Type Metrics <https://arxiv.org/abs/2007.12891>`_.
Moreover, we note that for the second Lamé parameter :math:`\mu`, cashocs implements
an idea from `Schulz and Siebenborn, Computational Comparison of Surface Metric for PDE Constrained Shape Optimization
<https://doi.org/10.1515/cmam-2016-0009>`_: There, it is proposed to compute :math:`\mu`
as the solution of the Laplace problem

.. math::
    \begin{alignedat}{2}
        - \Delta \mu &= 0 \quad &&\text{ in } \Omega, \\
        \mu &= \mu_\text{def} \quad &&\text{ on } \Gamma^\text{def},\\
        \mu &= \mu_\text{fix} \quad &&\text{ on } \Gamma^\text{fix}.\\
    \end{alignedat}

This allows to give the deformable and fixed boundaries a different stiffness,
which is then smoothly extended into the interior of the domain. Moreover, they
propose to use the solution of this Laplace equation directly for 2D problems,
and to use :math:`\sqrt{\mu}` for 3D problems.

Moreover, let us take a look at the possible _typing of boundaries that can be used
with cashocs. In principle, there exist
two _typing: deformable and fixed boundaries. On fixed boundaries, we
impose homogeneous Dirichlet boundary conditions for the shape gradient, so that
these are not moved under the corresponding deformation. In cashocs, we define what boundaries
are fixed and deformable via their markers, which are either defined in the
corresponding python script, or in the GMSH file, if such a mesh is imported.

The config file for :ref:`demo_shape_poisson` defines the deformable boundaries
with the command 

.. code-block:: ini

    shape_bdry_def = [1]

.. note::

    Remember, that in :ref:`demo_shape_poisson`, we defined :python:`boundaries` with the commands

    .. code-block:: ini

        boundary = CompiledSubDomain('on_boundary')
        boundaries = MeshFunction('size_t', mesh, dim=1)
        boundary.mark(boundaries, 1)

    Hence, we see that the marker :python:`1` corresponds to the entire boundary, so that this
    is set to being deformable through the config.

As we do not have a fixed boundary for this problem, the corresponding list
for the fixed boundaries is empty 

.. code-block:: ini

    shape_bdry_fix = []

Note, that cashocs also gives you the possibility of defining partially constrainted
boundaries, where only one axial component is fixed, whereas the other two are
not. These are defined in 

.. code-block:: ini

    shape_bdry_fix_x = []
    shape_bdry_fix_y = []
    shape_bdry_fix_z = []

For these, we have that :ini:`shape_bdry_fix_x` is a list of all markers whose corresponding
boundaries should not be deformable in x-direction, but can be deformed in the y-
and z-directions. Of course you can constrain a boundary to be only variable in a
single direction by adding the markers to the remaining lists.

Furthermore, we have the parameter :ini:`fixed_dimensions`, which enables us to restrict the shape gradient to specific dimensions. It is set via 

.. code-block:: ini

    fixed_dimensions = []

In case :ini:`fixed_dimensions = []`, there is no restriction on the shape gradient. However, if :ini:`fixed_dimensions = [i]`, then the i-th component of the shape gradient is set to 0, so that we have no deformation in the i-th coordinate direction. For example, if :ini:`fixed_dimensions = [0, 2]`, we only have a deformation in the :math:`y`-component of the mesh. The default is :ini:`fixed_dimensions = []`.

The next parameter is specified via

.. code-block:: ini

    use_pull_back = True

This parameter is used to determine, whether the material derivative should
be computed for objects that are not state or adjoint variables. This is
enabled by default.

.. warning::

    This parameter should always be set to :ini:`use_pull_back = True`, otherwise the shape derivative might
    be wrong. Only disable it when you are sure what you are doing.

    Furthermore, note that the material derivative computation is only correct,
    as long as no differential operators act on objects that are not state or
    adjoint variables. However, this should only be a minor restriction and not
    relevant for almost all problems.

.. note::

    See :ref:`demo_inverse_tomography` for a case, where we use
    :ini:`use_pull_back = False`.

The next parameters determine the coefficients of the bilinear form :math:`a`.
First, we have the first Lamé parameter :math:`\lambda`, which is set via 

.. code-block:: ini

    lambda_lame = 1.428571428571429

The default value for this is :ini:`lambda_lame = 0.0`.

Next, we specify the damping parameter :math:`\delta` with the line

.. code-block:: ini

    damping_factor = 0.2

The default for this is :ini:`damping_factor = 0.0`.

.. note::

    As the default value for the damping factor is :ini:`damping_factor = 0.0`, this
    should be set to a positive value in case the entire boundary of a problem
    is deformable. Otherwise, the Riesz identification problem for the shape
    gradient is not well-posed.

Finally, we define the values for :math:`\mu_\text{def}` and :math:`\mu_\text{fix}`
via

.. code-block:: ini

    mu_fix = 0.35714285714285715
    mu_def = 0.35714285714285715

The default behavior is given by :ini:`mu_fix = 1.0` and :ini:`mu_def = 1.0`.

The parameter :ini:`use_sqrt_mu` is a boolean flag, which switches between using
:math:`\mu` and :math:`\sqrt{\mu}` as the stiffness for the linear elasticity
equations, as discussed above. This is set via 

.. code-block:: ini

    use_sqrt_mu = False

and the default value is :ini:`use_sqrt_mu = False`.

The next line in the config file is

.. code-block:: ini

    inhomogeneous = False

This determines, whether an inhomogeneous linear elasticity equation is used to
project the shape gradient. This scales the parameters :math:`\mu, \lambda` and
:math:`\delta` by :math:`\frac{1}{\text{vol}}`, where :math:`\text{vol}` is the
volume of the current element (during assembly). This means, that smaller elements
get a higher stiffness, so that the deformation takes place in the larger elements,
which can handle larger deformations without reducing their quality too much. For
more details on this approach, we refer to the paper `Blauth, Leithäuser, and Pinnau,
Model Hierarchy for the Shape Optimization of a Microchannel Cooling System
<https://doi.org/10.1002/zamm.202000166>`_.

Moreover, the parameter 

.. code-block:: ini

    update_inhomogeneous = False

can be used to update the local mesh size after each mesh deformation, in case this is :ini:`update_inhomogeneous = True`, so that elements which become smaller also obtain a higher stiffness and vice versa. The default is :ini:`update_inhomogeneous = False`.

For the inhomogeneous mesh stiffness, we also have the parameter :ini:`inhomogeneous_exponent`, which is specified via

.. code-block:: ini

    inhomogeneous_exponent = 1.0

This parameter can be used to specify an exponent for the inhomogeneous mesh stiffness, so that the parameters
:math:`\mu, \lambda` and :math:`\delta` are scaled by :math:`\left( \frac{1}{\text{vol}} \right)^p`, where
:math:`p` is specified in :ini:`inhomogeneous_exponent`. The default for this parameter is :ini:`inhomogeneous_exponent = 1.0`.

There is also a different possibility to define the stiffness parameter :math:`\mu`
using cashocs, namely to define :math:`\mu` in terms of how close a point of the
computational domain is to a boundary. In the following we will explain this
alternative way of defining :math:`\mu`.
To do so, we must first set the boolean parameter

.. code-block:: ini

    use_distance_mu = True

which enables this formulation and deactivates the previous one. Note that by default,
the value is :ini:`use_distance_mu = False`. Next, we have the parameters :ini:`dist_min`, :ini:`dist_max`,
:ini:`mu_min` and :ini:`mu_max`. These do the following: If the distance to the boundary is
smaller than :ini:`dist_min`, the value of :math:`\mu` is set to :ini:`mu_min`, and if the distance
to the boundary is larger than :ini:`dist_max`, :math:`\mu` is set to :ini:`mu_max`. If the distance
to the boundary is between :ini:`dist_min` and :ini:`dist_max`, the value of :math:`\mu` is
interpolated between :ini:`mu_min` and :ini:`mu_max`. The type of this interpolation is
determined by the parameter 

.. code-block:: ini

    smooth_mu = True

If this parameter is set to :ini:`smooth_mu = True`, then a smooth, cubic polynomial is used to
interplate between :ini:`mu_min` and :ini:`mu_max`, which yields a continuously differentiable
:math:`\mu`. If this is set to :ini:`smooth_mu = False`, then a linear interpolation is used, which only yields
a continuous :math:`\mu`. The default for this parameter is :ini:`smooth_mu = False`.

Finally, we can specify which boundaries we want to incorporate when computing the
distance. To do so, we can specify a list of indices which contain the boundary
markers in the parameter

.. code-block:: ini

    boundaries_dist = [1,2,3]

This means, that only boundaries marked with 1, 2, and 3 are considered for computing
the distance, and all others are ignored. The default behavior is that all (outer) boundaries
are considered.

There is also another possibility to compute the shape gradient in cashocs, namely using the :math:`p`-Laplacian, as proposed by `Müller, Kühl, Siebenborn, Deckelnick, Hinze, and Rung <https://doi.org/10.1007/s00158-021-03030-x>`_. In order to do so, we have the following line

.. code-block:: ini

   use_p_laplacian = False

If this is set to :ini:`use_p_laplacian = True`, the :math:`p`-Laplacian is used to compute the shape gradient, as explained in :ref:`demo_p_laplacian`. However, by default this is disabled.
The value of :math:`p` which is then used is defined in the next line

.. code-block:: ini

    p_laplacian_power = 6

which defaults to :ini:`p_laplacian_power = 2`, whenever the parameter is not defined. The higher :math:`p` is chosen, the better the numerical are expected to be, but the numerical solution of the problem becomes more involved.

Finally, there is the possibility to use a stabilized weak form for the :math:`p`-Laplacian operator, where the stabilization parameter can be defined in the line

.. code-block:: ini

    p_laplacian_stabilization = 0.0

The default value of this parameter is :ini:`p_laplacian_stabilization = 0.0`. Note, that the parameter should be chosen comparatively small, i.e., significantly smaller than 1.

Moreover, we have the parameter :ini:`degree_estimation` which is specified via

.. code-block:: ini

    degree_estimation = True

This parameter enables cashocs' default estimation of the quadrature degree for the shape derivative. If this is set to `False`, an error related to FEniCS may occur - so this should be always enabled.

Next, we have the parameter :ini:`global_deformation` which is set via the line

.. code-block:: ini

    global_deformation = False

If this is set to `True`, cashocs computes the deformation from the initial to the optimized mesh (even when remeshing has been performed). This can, however, lead to some unexpected errors with PETSc, so this should be used with care.

We have the parameter :ini:`test_for_intersections`, which is specified via

.. code-block:: ini

    test_for_intersections = True

If this parameter is set to `True`, cashocs will check the deformed meshes for (self) intersections, which would generate non-physical geometries and reject them - so that all generated designs are physically meaningful. This should not be set to `False`.


.. _config_shape_regularization:

Section Regularization
----------------------

In this section, the parameters for shape regularizations are specified. For a
detailed discussion of their usage, we refer to :ref:`demo_regularization`.

First, we have the parameters :ini:`factor_volume` and :ini:`target_volume`. These are set
via the lines

.. code-block:: ini

    factor_volume = 0.0
    target_volume = 3.14

They are used to implement the (target) volume regularization term

.. math::

    \frac{\mu_\text{vol}}{2} \left( \int_{\Omega} 1 \text{ d}x - \text{vol}_\text{des} \right)^2

Here, :math:`\mu_\text{vol}` is specified via :ini:`factor_volume`, and :math:`\text{vol}_\text{des}`
is the target volume, specified via :ini:`target_volume`. The default behavior is
:ini:`factor_volume = 0.0` and :ini:`target_volume = 0.0`, so that we do not have
a volume regularization.

The next line, i.e.,

.. code-block:: ini

    use_initial_volume = True

determines the boolean flag :ini:`use_initial_volume`. If this is set to :ini:`use_initial_volume = True`,
then not the value given in :ini:`target_volume` is used, but instead the
volume of the initial geometry is used for :math:`\text{vol}_\text{des}`.

For the next two _typing of regularization, namely the (target) surface and (target)
barycenter regularization, the syntax for specifying the parameters is completely
analogous. For the (target) surface regularization we have

.. code-block:: ini

    factor_surface = 0.0
    target_surface = 1.0

These parameter are used to implement the regularization term

.. math::

    \frac{\mu_\text{surf}}{2} \left( \int_{\Gamma} 1 \text{ d}s - \text{surf}_\text{des} \right)^2

Here, :math:`\mu_\text{surf}` is determined via :ini:`factor_surface`, and
:math:`\text{surf}_\text{des}` is determined via :ini:`target_surface`. The default
values are given by :ini:`factor_surface = 0.0` and :ini:`target_surface = 0.0`.

As for the volume regularization, the parameter

.. code-block:: ini

    use_initial_surface = True

determines whether the target surface area is specified via :ini:`target_surface`
or if the surface area of the initial geometry should be used instead. The default
behavior is given by :ini:`use_initial_surface = False`.

Next, we have the curvature regularization, which is controlled by the parameter

.. code-block:: ini

    factor_curvature = 0.0

This is used to determine the size of :math:`\mu_\text{curv}` in the regularization
term

.. math::

    \frac{\mu_\text{curv}}{2} \int_{\Gamma} \kappa^2 \text{ d}s,

where :math:`\kappa` denotes the mean curvature. This regularization term can be
used to generate more smooth boundaries and to prevent kinks from occurring.

Finally, we have the (target) barycenter regularization. This is specified via
the parameters

.. code-block:: ini

    factor_barycenter = 0.0
    target_barycenter = [0.0, 0.0, 0.0]

and implements the term

.. math::

    \frac{\mu_\text{bary}}{2} \left\lvert \frac{1}{\text{vol}(\Omega)} \int_\Omega x \text{ d}x - \text{bary}_\text{des} \right\rvert^2

The default behavior is given by :ini:`factor_barycenter = 0.0` and :ini:`target_barycenter = [0,0,0]`,
so that we do not have a barycenter regularization.

The flag

.. code-block:: ini

    use_initial_barycenter = True

again determines, whether :math:`\text{bary}_\text{des}` is determined via :ini:`target_barycenter`
or if the barycenter of the initial geometry should be used instead. The default behavior
is given by :ini:`use_initial_barycenter = False`.

.. hint::

    The object :ini:`target_barycenter` has to be a list. For 2D problems it is also
    sufficient, if the list only has two entries, for the :math:`x` and :math:`y`
    barycenters.

Finally, we have the parameter :ini:`use_relative_scaling` which is set in the line 

.. code-block:: ini

    use_relative_scaling = False

This boolean flag does the following. For some regularization term :math:`J_\text{reg}(\Omega)` with corresponding
factor :math:`\mu` (as defined above), the default behavior is given by :ini:`use_relative_scaling = False`
adds the term :math:`\mu J_\text{reg}(\Omega)` to the cost functional, so that the
factor specified in the configuration file is actually used as the factor for the regularization term.
In case :ini:`use_relative_scaling = True`, the behavior is different, and the following term is
added to the cost functional: :math:`\frac{\mu}{\left\lvert J_\text{reg}(\Omega_0) \right\rvert} J_\text{reg}(\Omega)`,
where :math:`\Omega_0` is the initial guess for the geometry. In particular, this means
that the magnitude of the regularization term is equal to :math:`\mu` on the initial geometry.
This allows a detailed weighting of multiple regularization terms, which is particularly
useful in case the cost functional is also scaled (see :ref:`demo_scaling`).

.. _config_shape_mesh_quality:

Section MeshQuality
-------------------

This section details the parameters that influence the quality of the
computational mesh. First, we have the lines

.. code-block:: ini

    volume_change = inf
    angle_change = inf

These parameters are used to specify how much the volume and the angles, respectively,
of the mesh elements are allowed to change in a single transformation. In particular,
they implement the following criteria (see `Etling, Herzog, Loayza, Wachsmuth,
First and Second Order Shape Optimization Based on Restricted Mesh Deformations
<https://doi.org/10.1137/19M1241465>`_)

.. math::

    \frac{1}{\alpha} &\leq \det\left( \text{id} + D\mathcal{V} \right) \leq \alpha \\
    \left\lvert\left\lvert D\mathcal{V} \right\rvert\right\rvert_{F} &\leq \beta.

Here, :math:`\alpha` corresponds to :ini:`volume_change` and :math:`\beta` corresponds
to :ini:`angle_change`, and :math:`\mathcal{V}` is the deformation. The default behavior
is given by :ini:`volume_change = inf` and :ini:`angle_change = inf`, so that no restrictions
are posed. Note, that, e.g., `Etling, Herzog, Loayza, Wachsmuth,
First and Second Order Shape Optimization Based on Restricted Mesh Deformations
<https://doi.org/10.1137/19M1241465>`_ use the values :ini:`volume_change = 2.0` and
:ini:`angle_change = 0.3`.

The next two parameters are given byx

.. code-block:: ini

    tol_lower = 0.0
    tol_upper = 1e-15

These parameters specify a kind of interval for the mesh quality. In particular,
we have the following situation (note that the mesh quality is always an element
in :math:`[0,1]`):

- If the mesh quality is in :math:`[\texttt{tol upper}, 1]`, the mesh is assumed
  to be "good", so that finite element solutions of the corresponding PDEs are
  sensible and not influenced by the mesh quality or discretization artifacts.

- If the mesh quality is in :math:`[\texttt{tol lower}, \texttt{tol upper}]`, a
  kind of breaking point is reached. Here, it is assumed that the mesh is sufficiently
  good so that the solution of the state system is still possible. However, a mesh
  whose quality is in this interval should not be used anymore to compute the solution
  of the adjoint system or to compute the shape gradient, as the quality is too poor
  for this purpose. Usually, this means that the algorithm is terminated, unless remeshing
  is enabled. In the latter case, remeshing is performed.

- If the mesh quality is in the interval :math:`[0, \texttt{tol lower}]`, the mesh
  quality is assumed to be so poor, that even the solution of the state system
  is not possible anymore. In practice, this can only happen during the Armijo line
  search. Thanks to our previous considerations, we also know that the mesh, that is
  to be deformed, has at least a quality of :ini:`tol_lupper`, so that this quality
  might be reached again, if the step size is just decreased sufficiently often.
  This way, it is ensured that the state system is only solved when the mesh quality
  is larger than :ini:`tol_lower`, so that the corresponding cost functional value is
  reasonable.

The default behavior is given by :ini:`tol_lower = 0.0` and :ini:`tol_upper = 1e-15`,
so that there are basically no requirements on the mesh quality.

Finally, the upcoming two parameters specify how exactly the mesh quality is measured.
The first one is

.. code-block:: ini

    measure = condition_number

and determines one of the four mesh quality criteria, as defined in :py:class:`MeshQuality <cashocs.MeshQuality>`.
Available options are

- :ini:`measure = skewness`
- :ini:`measure = maximum_angle`
- :ini:`measure = radius_ratios`
- :ini:`measure = condition_number`

(see :py:class:`MeshQuality <cashocs.MeshQuality>` for a detailed description).
The default value is given by :ini:`measure = skewness`.

The parameter :ini:`type` determines, whether the minimum quality over all
elements (:ini:`type = min`) or the average quality over all elements (:ini:`type = avg`)
shall be used. This is set via 

.. code-block:: ini

    type = min

and defaults to :ini:`type = min`.

Finally, we have the parameter :ini:`remesh_iter` in which the user can specify after how many iterations a remeshing should be performed. It is given by

.. code-block:: ini

    remesh_iter = 0

where :ini:`remesh_iter = 0` means that no automatic remeshing is performed (this is the default), and :ini:`remesh_iter = n` means that remeshing is performed after each `n` iterations. Note that to use this parameter and avoid unexpected results, it might be beneficial to the the lower and upper mesh quality tolerances to a low value, so that the "quality based remeshing" does not interfere with the "iteration based remeshing", but both can be used in combination.

.. _config_shape_mesh_quality_constraints:

Section MeshQualityConstraints
------------------------------

The parameter :ini:`min_angle` is used to define the threshold angle, i.e. the minimum (solid) angle which is feasible for the mesh. The default is given by

.. code-block:: ini

	min_angle = 0.0

which ensures that the constraints are not active by default. Note that the initial mesh has to be feasible for the method to work, so if the minimum angle in the mesh is smaller than the :ini:`min_angle` specified in the configuration, cashocs will raise an exception. Note that the angle is specified in radian and **not** degree.

To circumvent this problem for meshes with small angles (which could be used, e.g., to resolve boundary layers, the next parameter :ini:`feasible_angle_reduction_factor` is used. This parameter specifies, how much smaller the (solid) angles of the mesh are allowed to become relative to the value in the initial mesh. That means a value of :ini:`feasible_angle_reduction_factor = 0.25` ensures that no (solid) angle in a mesh element will become smaller than one quarter of the smallest angle of the element in the initial mesh. The default is given by

.. code-block:: ini

	feasible_angle_reduction_factor = 0.0

which ensures that the constraints are not active by default.

.. note::

	If both the :ini:`feasible_angle_reduction_factor` and :ini:`min_angle` are given, cashocs uses the element-wise minimum of the two. In particular, this means that a strategy of using :ini:`feasible_angle_reduction_factor = 0.9999` and some value for :ini:`min_angle` can be used to constrain the (solid) angle to a specific value, wherever this is possible (and leave the angles that are below this threshold as they are).

The parameter :ini:`tol` is used to define a tolerance for which constraints are treated as active or not. As we treat the constraints numerically, they can only be satisfied up to a certain tolerance, which the user can specify here. The default value of

.. code-block:: ini

	tol = 1e-2

should work well in most situations. In some situations, the optimization could be faster with a tolerance of :ini:`tol = 1e-1` (but should never be larger) or more accurate when using, e.g., :ini:`tol = 1e-3` (lower values should most of the time not be necessary).

The parameter :ini:`mode` can only be set to

.. code-block:: ini

	mode = approximate

at the moment, which is also the default value. In the future, other options might be possible.


.. _config_shape_output:

Section Output
--------------

In this section, the parameters for the output of the algorithm, either in the terminal
or as files, are specified. First, we have the parameter :ini:`verbose`. This is used to toggle the output of the
optimization algorithm. It defaults to :ini:`verbose = True` and is controlled via

.. code-block:: ini

    verbose = True

The parameter :ini:`save_results` is a boolean flag, which determines whether a history
of the optimization algorithm, including cost functional value, gradient norm, accepted
step sizes, and mesh quality, shall be saved to a .json file. This defaults to :ini:`save_results = True`,
and can be set with

.. code-block:: ini

    save_results = False

Moreover, we define the parameter :ini:`save_txt`

.. code-block:: ini

	save_txt = False

This saves the output of the optimization, which is usually shown in the terminal,
to a .txt file, which is human-readable.

The next line in the config file is

.. code-block:: ini

    save_state = False

Here, the parameter :ini:`save_state` is set. This is a boolean flag, which can be set to
:ini:`save_state = True` to enable that cashocs generates .xdmf files for the state variables for each iteration the optimization algorithm performs. These are great for visualizing the
steps done by the optimization algorithm, but also need some disc space, so that they are disabled by default.
Note, that for visualizing these files, you need `Paraview <https://www.paraview.org/>`_.

The next parameter, :ini:`save_adjoint` works analogously, and is given in the line

.. code-block:: ini

    save_adjoint = False

If this is set to True, cashocs generates .xdmf files for the adjoint variables in each iteration of the optimization algorithm.
Its main purpose is for debugging.

The next parameter is given by :ini:`save_gradient`, which is given in the line

.. code-block:: ini

    save_gradient = False

This boolean flag ensures that a paraview with the computed shape gradient is saved in ``result_dir/xdmf``. The main purpose of this is for debugging.

Moreover, we also have the parameter :ini:`save_mesh` that is set via

.. code-block:: ini

    save_mesh = False

This is used to save the mesh as a GMSH file in each iteration of the optimization. The default behavior
is given by :ini:`save_mesh = False`. Note, that this is only
possible if the input mesh was already generated by GMSH, and specified in :ref:`the Mesh
section of the config file <config_shape_mesh>`. For any other meshes, the underlying mesh is also saved in
the .xdmf files, so that you can at least always visualize the optimized geometry.

In the end, we also have, like for optimal control problems, a parameter that specifies
where the output is placed, again named :ini:`result_dir`, which is given in the config file
in the line 

.. code-block:: ini

    result_dir = ./results

As before, this is either a relative or absolute path to the directory where the
results should be placed.

The parameter :ini:`precision`, which is set via

.. code-block:: ini

    precision = 3

is an integer parameter which determines how many significant digits are printed in the output to the console and / or the result file.

Moreover, we have the parameter :ini:`time_suffix`, which adds a suffix to the result directory based on the current time. It is controlled by the line

.. code-block:: ini

	time_suffix = False



.. _config_shape_summary:

Summary
-------

Finally, an overview over all parameters and their default values can be found
in the following.


[Mesh]
******

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`mesh_file`
      - Only needed for remeshing
    * - :ini:`gmsh_file`
      - Only needed for remeshing
    * - :ini:`geo_file`
      - Only needed for remeshing
    * - :ini:`remesh = False`
      - if :ini:`remesh = True`, remeshing is enabled; this feature is experimental, use with care
    * - :ini:`show_gmsh_output = False`
      - if :ini:`show_gmsh_output = True`, shows the output of GMSH during remeshing in the console



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
    * - :ini:`method = armijo`
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



[ShapeGradient]
***************

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`shape_bdry_def = []`
      - list of indices for the deformable boundaries
    * - :ini:`shape_bdry_fix = []`
      - list of indices for the fixed boundaries
    * - :ini:`shape_bdry_fix_x = []`
      - list of indices for boundaries with fixed x values
    * - :ini:`shape_bdry_fix_y = []`
      - list of indices for boundaries with fixed y values
    * - :ini:`shape_bdry_fix_z = []`
      - list of indices for boundaries with fixed z values
    * - :ini:`fixed_dimensions = []`
      - a list of coordinates which should be fixed during the shape optimization (x=0, y=1, etc.)
    * - :ini:`use_pull_back = True`
      - if :ini:`use_pull_back = False`, shape derivative might be wrong; no pull-back for the material derivative is performed;
        only use with caution
    * - :ini:`lambda_lame = 0.0`
      - value of the first Lamé parameter for the elasticity equations
    * - :ini:`damping_factor = 0.0`
      - value of the damping parameter for the elasticity equations
    * - :ini:`mu_def = 1.0`
      - value of the second Lamé parameter on the deformable boundaries
    * - :ini:`mu_fix = 1.0`
      - value of the second Lamé parameter on the fixed boundaries
    * - :ini:`use_sqrt_mu = False`
      - if :ini:`use_sqrt_mu = True`, uses the square root of the computed ``mu_lame``; might be good for 3D problems
    * - :ini:`inhomogeneous = False`
      - if :ini:`inhomogeneous = True`, uses inhomogeneous elasticity equations, weighted by the local mesh size
    * - :ini:`update_inhomogeneous = False`
      - if :ini:`update_inhomogeneous = True` and :ini:`inhomogeneous=True`, then the weighting with the local mesh size is updated as the mesh is deformed.
    * - :ini:`inhomogeneous_exponent = 1.0`
      - The exponent for the inhomogeneous mesh stiffness
    * - :ini:`use_distance_mu = False`
      - if :ini:`use_distance_mu = True`, the value of the second Lamé parameter is computed via the distance to the boundary
    * - :ini:`dist_min = 1.0`
      - Specifies the distance to the boundary, until which :math:`\mu` is given by :ini:`mu_min`
    * - :ini:`dist_max = 1.0`
      - Specifies the distance to the boundary, until which :math:`\mu` is given by :ini:`mu_max`
    * - :ini:`mu_min = 1.0`
      - The value of :math:`\mu` for a boundary distance smaller than :ini:`dist_min`
    * - :ini:`mu_max = 1.0`
      - The value of :math:`\mu` for a boundary distance larger than :ini:`dist_max`
    * - :ini:`boundaries_dist = []`
      - The indices of the boundaries, which shall be used to compute the distance, :ini:`boundaries_dist = []` means that all boundaries are considered
    * - :ini:`smooth_mu = False`
      - If false, a linear (continuous) interpolation between :ini:`mu_min` and :ini:`mu_max` is used, otherwise a cubic :math:`C^1` interpolant is used
    * - :ini:`use_p_laplacian = False`
      - If :ini:`use_p_laplacian = True`, then the :math:`p`-Laplacian is used to compute the shape gradient
    * - :ini:`p_laplacian_power = 2`
      - The parameter :math:`p` of the :math:`p`-Laplacian
    * - :ini:`p_laplacian_stabilization = 0.0`
      - The stabilization parameter for the :math:`p`-Laplacian problem. No stabilization is used when this is :ini:`p_laplacian_stabilization = 0.0`.
    * - :ini:`global_deformation = False`
      - Computes the global deformation from initial to optimized mesh. This can lead to unexpected errors, use with care.
    * - :ini:`degree_estimation = True`
      - Estimate the required degree for quadrature of the shape derivative. This should be `True`, otherwise unexpected errors can happen.
    * - :ini:`test_for_intersections = True`
      - If enabled, the mesh is tested for intersections which would create non-physical meshes. This should always be enabled, otherwise the obtained results might be incorrect.


[Regularization]
****************

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`factor_volume = 0.0`
      - value of the regularization parameter for volume regularization; needs to be non-negative
    * - :ini:`target_volume = 0.0`
      - prescribed volume for the volume regularization
    * - :ini:`use_initial_volume = False`
      - if :ini:`use_initial_volume = True` uses the volume of the initial geometry as prescribed volume
    * - :ini:`factor_surface = 0.0`
      - value of the regularization parameter for surface regularization; needs to be non-negative
    * - :ini:`target_surface = 0.0`
      - prescribed surface for the surface regularization
    * - :ini:`use_initial_surface = False`
      - if :ini:`use_initial_surface = True` uses the surface area of the initial geometry as prescribed surface
    * - :ini:`factor_curvature = 0.0`
      - value of the regularization parameter for curvature regularization; needs to be non-negative
    * - :ini:`factor_barycenter = 0.0`
      - value of the regularization parameter for barycenter regularization; needs to be non-negative
    * - :ini:`target_barycenter = [0.0, 0.0, 0.0]`
      - prescribed barycenter for the barycenter regularization
    * - :ini:`use_initial_barycenter = False`
      - if :ini:`use_initial_barycenter = True` uses the barycenter of the initial geometry as prescribed barycenter
    * - :ini:`use_relative_scaling = False`
      - if :ini:`use_relative_scaling = True`, the regularization terms are scaling so that they have the magnitude specified in the respective factor for the initial iteration.



[MeshQuality]
*************

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`volume_change = inf`
      - determines by what factor the volume of a cell is allowed to change within a single deformation
    * - :ini:`angle_change = inf`
      - determines how much the angles of a cell are allowed to change within a single deformation
    * - :ini:`tol_lower = 0.0`
      - if the mesh quality is lower than this tolerance, the state system is not solved
        for the Armijo rule, instead step size is decreased
    * - :ini:`tol_upper = 1e-15`
      - if the mesh quality is between :ini:`tol_lower` and :ini:`tol_upper`, the state
        system will still be solved for the Armijo rule. If the accepted step yields a quality
        lower than this, algorithm is terminated (or remeshing is initiated)
    * - :ini:`measure = skewness`
      - determines which quality measure is used
    * - :ini:`type = min`
      - determines if minimal or average quality is considered
    * - :ini:`remesh_iter`
      - When set to a value > 0, remeshing is performed every :ini:`remesh_iter` iterations.


[MeshQualityConstraints]
************************

.. list-table::
    :header-rows: 1

    * - Parameter = Default value
      - Remarks
    * - :ini:`min_angle = 0.0`
      - The minimum feasible triangle / solid angle of the mesh cells in radian. This is constant for all cells. If this is positive, the constraints are used. If this is 0, no constraints are used.
    * - :ini:`tol = 1e-2`
      - The tolerance for the mesh quality constraints. If `abs(g(x)) < tol`, then the constraint is considered active
    * - :ini:`mode = approximate`
      - The mode for calculating the (shape) derivatives of the constraint functions. At the moment, only "approximate" is supported.
    * - :ini:`feasible_angle_reduction_factor = 0.0`
      - A factor in the interval [0,1) which sets the feasible reduction of the triangle / solid angles. This means, that each cell is only allowed to have angles larger than this times the initial minimum angle.

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
      - if :ini:`save_gradient = True`, the history of the shape gradient over the optimization is saved in .xdmf files
    * - :ini:`save_mesh = False`
      - if :ini:`save_mesh = True`, saves the mesh in each iteration of the optimization; only available for GMSH input
    * - :ini:`result_dir = ./results`
      - path to the directory, where the output should be placed
    * - :ini:`precision = 3`
      - number of significant digits to be printed
    * - :ini:`time_suffix = False`
      - Boolean flag, which adds a suffix to :ini:`result_dir` based on the current time
