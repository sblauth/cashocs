.. _config_shape_optimization:

Documentation of the Config Files for Shape Optimization Problems
=================================================================

Let us take a detailed look at the config files for shape optimization problems and
discusss the corresponding parameters and the possible choices. The corresponding
config file used for this discussion is :download:`config.ini </../../demos/documented/shape_optimization/shape_poisson/config.ini>`.

For the shape optimization, the config file is a lot larger compared to the :ref:`config files
for optimal control <config_optimal_control>`.
However, the most important parameters are shared between both types of optimization
problems.

A general config file for shape optimization has a total of 6 sections, namely
:ref:`Mesh <config_shape_mesh>`, :ref:`StateEquation <config_shape_state_equation>`,
:ref:`OptimizationRoutine <config_shape_optimization_routine>`, :ref:`ShapeGradient <config_shape_shape_gradient>`,
:ref:`Regularization <config_shape_regularization>`, and :ref:`MeshQuality <config_shape_mesh_quality>`. We go over these
sections and each parameter in them in the following.



.. _config_shape_mesh:

Section Mesh
------------

The mesh section is, in contrast to the corresponding one for optimal control problems,
more important and also has more parameters. However, its primary importance is for
remeshing, which we cover ind detail in another demo.

As first parameter, we have ::

    mesh_file = ./mesh/mesh.xdmf

This specifies a path to a .xdmf file containing the discretized geometry. For all purposes, CASHOCS assumes that this .xdmf file was generated via conversion from a
gmsh file using the command line command ::

    cashocs-convert in.msh out.xdmf

Note, that the corresponding files for the boundaries and subdomains are generated automatically with ``cashocs-convert``, and they will also be read if they are present.
This applies to both uses of :py:func:`import_mesh <cashocs.import_mesh>`, so passing either the config file
or the path to the (main) .xdmf mesh file works here. Note, that the parameter ``mesh_file`` is only important for remeshing. For all other cases, the mesh can be imported / generated in whatever way you prefer, but it might not be possible to
read these files successfully with :py:func:`import_mesh <cashocs.import_mesh>`.

The second parameter in the Mesh section, ``gmsh_file``, is defined via ::

    gmsh_file = ./mesh/mesh.msh

This defines the path to the gmsh .msh file which was used to create the .xdmf_file
specified in ``mesh_file``. As before, this parameter is only relevant for remeshing
purposes, and not needed otherwise.

The next parameter is ``geo_file``, which is the final file we need for remeshing (
and only there). It is also given by a path to a file, in this case to the gmsh .geo
file used to generate the ``gmsh_file``. It is specified, .e.g., as ::

    geo_file = ./mesh/mesh.geo

Next up is a boolean flag that is used to indicate, whether remeshing shall be performed ::

    remesh = False


As the remeshing feature is somewhat experimental, we do advise to always try without
remeshing. Note, that by default this flag is set to ``False`` and remeshing is disabled.

Finally, we have the boolean flag ``show_gmsh_output``, specified via ::

    show_gmsh_output = False

This is used to toggle on / off the command line output of gmsh when it performs a
remeshing operation. This can be helpful for debugging purposes.

As stated throughout the Mesh section, these parameters are optional most of the time,
and only really required for remeshing. You can safely leave them out of your config file, and you should not need them, unless you want to perform remeshing.


.. _config_shape_state_equation:

Section StateEquation
---------------------

The StateEquation section is in complete analogy to :ref:`the corresponding one for optimal control problems <config_ocp_state_equation>`. For the
sake of completeness, we briefly recall the parameters here, anyway.

The first parameter is ``is_linear``, and can be set as ::

    is_linear = True

This is a boolean flag that indicates whether the state system is linear or not. The default value for this parameter is ``False``, as every linear problem can also be
interpreted as a nonlinear one. In this case, the Newton method converges after a single
iteration.

The next parameters are used to define the tolerances of the Newton solver, in
case a nonlinear state system has to be solved ::

    newton_atol = 1e-13
    newton_rtol = 1e-11

Here, ``atol`` sets the absolute, and ``rtol`` the relative tolerance.

The parameter ``newton_damped``, which is set via ::

    newton_damped = True

is a boolean flag, indicating whether a damping strategy should be performed for the
Newton method, or whether the classical Newton-Raphson iteration shall be used. This
defaults to True, but for some problems it might be beneficial (and faster) to not
use damping.

Next, we have the parameter ::

    newton_verbose = False

This is used to toggle the verbose output of the Newton method for the state system.
By default this is set to ``False`` so that there is not too much noise.

The final parameter for the Newton iteration is the maximum number of iterations it
is allowed to perform before the iteration is cancelled. This is controlled via ::

    newton_iter = 50

The upcoming parameters are used to define the behavior of a Picard solver, that
may be used if we have multiple variables (see :ref:`demo_picard_iteration` for optimal control).
This is used in case multiple state variables are defined and the corresponding system shall be solved via a Picard iteration. First,
we have a boolean flag, set via ::

    picard_iteration = False

which determines whether the Picard iteration is enabled or not. By default, it is not. The following two parameters determine, analogously to before, the tolerances for the
Picard iteration ::

    picard_rtol = 1e-10
    picard_atol = 1e-12

Note, that the tolerances of the Newton solver are automatically adjusted in case
a Picard iteration is performed as to enable a faster, inexact Picard iteration.
The amout of iterations for the Picard iteration are set with ::

    picard_iter = 10

Finally, we can enable verbose output of the Picard iteration with the following
boolean flag ::

    picard_verbose = False

which is set to ``False`` by default.


.. _config_shape_optimization_routine:

Section OptimizationRoutine
---------------------------

The section OptimizationRoutine also closely resembles :ref:`the one for optimal control
problems <config_ocp_optimization_routine>`. Again, we will take a brief look at all parameters here

The first parameter that can be controlled via the config file is ``algorithm``, which is
set via ::

    algorithm = lbfgs

There are three possible choices for this for shape optimization problems, namely

- ``'gd'`` or ``'gradient_descent'`` : A gradient descent method

- ``'cg'``, ``'conjugate_gradient'``, ``'ncg'``, ``'nonlinear_cg'`` : Nonlinear CG methods

- ``'lbfgs'`` or ``'bfgs'`` : limited memory BFGS method.


The next parameter is used to control the maximum number of iterations performed by
the optimization algorithm. It is set via ::

    maximum_iterations = 50

Thereafter, we specify the tolerances for the optimization algorithm with the parameters ::

    rtol = 5e-3
    atol = 0.0

Again, ``rtol`` denotes the relative, and ``atol`` the absolute tolerance.

Next up, we have the initial guess for the step size, which can be determined via ::

    step_initial = 1.0

The upcoming parameters are used for the Armijo rule ::

    epsilon_armijo = 1e-4
    beta_armijo = 2

and are used to verify that the condition

.. math:: J((I + t \mathcal{V})\Omega) \leq J(\Omega) + \varepsilon_{\text{Armijo}}\ t\ dJ(\Omega)[\mathcal{V}],

and if this is not satisfied, the stepsize is updated via :math:`t = \frac{t}{\beta_{\text{Armijo}}}`.

The following parameter, ``soft_exit``, is used as a boolean flag which determines how
the optimization algorithm is terminated in case it does not converge. If ``soft_exit = True``, then an
error message is printed, but code after the :py:meth:`solve <cashocs.ShapeOptimizationProblem.solve>` call of the
optimization problem will still be executed. However, when ``soft_exit = False``, CASHOCS
raises an exception and stops python. This is set via ::

    soft_exit = False

and is disabled by default.

Next up, we have the parameter ``verbose``. This is used to toggle the output of the
optimization algorithm. It defaults to ``True`` and is controlled via ::

    verbose = True

The parameter ``save_results`` is a boolean flag, which determines whether a history
of the optimization algorithm, including cost function value, gradient norm, accepted
step sizes, and mesh quality, shall be saved to a .json file. This defaults to ``True``,
and can be set with ::

    save_results = False

The next line in our example config file is ::

    save_pvd = False

Here, the parameter ``save_pvd`` is set. This is a boolean flag, which can be set to
``True`` to enable that CASHOCS generates .pvd files for the state variables for each iteration the optimization algorithm performs. These are great for visualizing the
steps done by the optimization algorithm, but also need some disc space, so that they are disabled by default. For visualizing these files, you need `Paraview <https://www.paraview.org/>`_.

Moreover, we also have the parameter ``save_mesh`` that is set via ::

    save_mesh = False

This is used to save the optimized geometry to a gmsh file. Note, that this is only
possible if the input mesh was already generated by gmsh, and specified in the Mesh section of the config file. For any other meshes, the underlying mesh is also saved in
the .pvd files, so that you can at least always visualize the optimized geometry.

Limited memory BFGS method
**************************

Next, we discuss the parameters relevant for the limited memory BFGS method. For details
regarding this method, we refer to `Schulz, Siebenborn, and Welker, Efficient PDE Constrained Shape Optimization Based on Steklov-Poincaré-Type Metrics
<https://doi.org/10.1137/15M1029369>`_, where the methods are introduced.

The first parameter, ``memory_vectors``, determines how large the storage of the BFGS method is. It is set via ::

    memory_vectors = 3

Usually, a higher storage leads to a better Hessian approximation, and thus to faster
convergence. However, this also leads to an increased memory usage. Typically, values
below 5 already work very well.

The other parameter for the BFGS method is ::

    use_bfgs_scaling = True

This determines, whether one should use a scaling of the initial Hessian approximation
(see `Nocedal and Wright, Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_). This is usually very beneficial
and should be kept enabled (which it is by default).


Nonlinear conjugate gradient methods
************************************

The following parameters are used to define the behavior of the nonlinear conjugate
gradient methods for shape optimization. For more details on this, we refer to the
preprint `Blauth, Nonlinear Conjugate Gradient Methods for PDE Constrained Shape
Optimization Based on Steklov-Poincaré-Type Metrics <https://arxiv.org/abs/2007.12891>`_.

First, we define which nonlinear CG method is used by ::

    cg_method = DY

determines which of the nonlinear cg methods shall be used. Available are

- ``FR`` : The Fletcher-Reeves method

- ``PR`` : The Polak-Ribiere method

- ``HS`` : The Hestenes-Stiefel method

- ``DY`` : The Dai-Yuan method

- ``HZ`` : The Hager-Zhang method

As for optimal control problems, the subsequent parameters are used to define the
restart behavior of the nonlinear CG methods. First, we have ::

    cg_periodic_restart = False

This boolean flag en- or disables that the NCG methods are restarted after a fixed
amount of iterations, which is specified via ::

    cg_periodic_its = 5

i.e., if ``cg_periodic_restart = True`` and ``cg_periodic_its = n``, then the NCG method
is restarted every ``n`` iterations.

Alternatively, there also exists a relative restart criterion (see `Nocedal and Wright,
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_), which can be enabled
via the boolean flag ``cg_relative_restart``, which is defined in the line ::

    cg_relative_restart = False

and the corresponding restart tolerance is set in ::

    cg_restart_tol = 0.5

Note, that ``cg_restart_tol`` should be in :math:`(0, 1)`, and measures how "orthogonal"
two subsequent gradients generated by the method are. If they are not "sufficiently
orthogonal", then the method is restarted with a gradient step.

.. _config_shape_shape_gradient:

Section ShapeGradient
---------------------

After we have specified the behavior of the solution algorithm, this section
is used to specify parameters relevant to the computation of the shape gradient.
Note, that by "shape gradient" we refer to the following object.

Let :math:`\mathcal{S} \subset \{ \Omega \vert \Omega \subset \mathbb{R}^d \}` be a
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
form is given as follows, in a general form, that is also implemented in CASHOCS

.. math:: a \colon H \times H; \quad a(\mathcal{W}, \mathcal{V}) = \int_\Omega \mu D\mathcal{W} : D\mathcal{V} + \lambda \text{div}(\mathcal{W}) \text{div}(\mathcal{V}) + \delta V \cdot W \text{d}x,

where :math:`H` is some suitable subspace of :math:`H^1(\Omega)^d`. The subspace property is needed
to include certain geometrical constraints of the shape optimization problem, which fix
certain boundaries, into the shape gradient. For a detailed description of this
setting we refer to the preprint `Blauth, Nonlinear Conjugate Gradient Methods for PDE
Constrained Shape Optimization Based on Steklov-Poincaré-Type Metrics <https://arxiv.org/abs/2007.12891>`_.

First of all, we define what kind of boundaries there are. In principle, there exist
two types, the deformable boundaries and fixed boundaries. On fixed boundaries, we
impose homogeneous Dirichlet boundary conditions for the shape gradient, so that
these are not moved under the corresponding deformation. In CASHOCS, we define what boundaries
are fixed and deformable via their markers, which are either defined in the
corresponding python script, or in the gmsh file, if such a mesh is imported.

The config file for :ref:`demo_shape_poisson` defines the deformable boundaries
with the command ::

    shape_bdry_def = [1]

Remember, that in the demo, we defined ``boundaries`` with the commands ::

    boundary = CompiledSubDomain('on_boundary')
    boundaries = MeshFunction('size_t', mesh, dim=1)
    boundary.mark(boundaries, 1)

Hence, we see that the marker ``1`` corresponds to the entire boundary, and this
is set to being variable / deformable.

As we do not have a fixed boundary for this problem, the corresponding list
for the fixed boundaries is empty ::

    shape_bdry_fix = []

Note, that CASHOCS also gives you the possibility of defining partially constrainted
boundaries, where only one axial component is fixed, whereas the other two are
not. These are defined in ::

    shape_bdry_fix_x = []
    shape_bdry_fix_y = []
    shape_bdry_fix_z = []

For these, we have that ``shape_bdry_fix_x`` is a list of all markers whose corresponding
boundaries should not be deformable in x-direction, but can be deformed in the y-
and z-directions. Of course you can constrain a boundary to be only variable in a
single direction by adding the markers to the remaining lists.

The next parameters determine the coefficients of the bilinear form :math:`a`.



.. _config_shape_regularization:

Section Regularization
----------------------




.. _config_shape_mesh_quality:

Section MeshQuality
-------------------
