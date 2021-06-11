.. _config_shape_optimization:

Documentation of the Config Files for Shape Optimization Problems
=================================================================

Let us take a detailed look at the config files for shape optimization problems and
discusss the corresponding parameters. The corresponding
config file used for this discussion is :download:`config.ini </../../demos/documented/shape_optimization/shape_poisson/config.ini>`,
which is the config file used for :ref:`demo_shape_poisson`.

For shape optimization problems, the config file is a lot larger compared to the :ref:`config files
for optimal control <config_optimal_control>`.
However, several important parameters are shared between both types of optimization
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
we remark that CASHOCS has a default behavior for almost all of these
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

As first parameter, we have ::

    mesh_file = ./mesh/mesh.xdmf

This specifies a path to a .xdmf file containing the discretized geometry. For all purposes, CASHOCS assumes that this .xdmf file was generated via conversion from a
GMSH file using the command line command :ref:`cashocs-convert <cashocs_convert>`.

Note, that the corresponding files for the boundaries and subdomains are generated
automatically with ``cashocs-convert``, and they will also be read by :py:func:`import_mesh <cashocs.import_mesh>`
if they are present.


The second parameter in the Mesh section, ``gmsh_file``, is defined via ::

    gmsh_file = ./mesh/mesh.msh

This defines the path to the GMSH .msh file which was used to create the .xdmf file
specified in ``mesh_file``. As before, this parameter is only relevant for remeshing
purposes, and not needed otherwise.

The next parameter is ``geo_file``, which is the final file we need for remeshing (
and only there). It is also given by a path to a file, in this case to the GMSH .geo
file used to generate the ``gmsh_file``. It is specified, .e.g., as ::

    geo_file = ./mesh/mesh.geo

.. note::

    For a detailed discussion of how to use these parameters we refer to :ref:`demo_remeshing`.

Next up is a boolean flag that is used to indicate whether remeshing shall be performed ::

    remesh = False


As the remeshing feature is experimental, we do advise to always try without
remeshing. Note, that by default this flag is set to ``False`` so that remeshing is disabled.

Finally, we have the boolean flag ``show_gmsh_output``, specified via ::

    show_gmsh_output = False

This is used to toggle on / off the terminal output of GMSH when it performs a
remeshing operation. This can be helpful for debugging purposes. By default, this
is set to ``False``.

As stated throughout the Mesh section, these parameters are optional most of the time,
and are only really required for remeshing. You can safely leave them out of your config file, and you should not need them, unless you want to perform remeshing.


.. _config_shape_state_equation:

Section StateSystem
---------------------

The StateSystem section is in complete analogy to :ref:`the corresponding one for optimal control problems <config_ocp_state_system>`. For the
sake of completeness, we briefly recall the parameters here, anyway.

The first parameter is ``is_linear``, and can be set as ::

    is_linear = True

This is a boolean flag that indicates whether the state system is linear or not.
The default value for this parameter is ``False``, as every linear problem can also be
interpreted as a nonlinear one.

The next parameters are used to define the tolerances of the Newton solver, in
case a nonlinear state system has to be solved ::

    newton_rtol = 1e-11
    newton_atol = 1e-13


Here, ``newton_rtol`` sets the relative, and ``newton_atol`` the absolute tolerance
for Newton's method. Their default values are ``newton_rtol = 1e-11`` and
``newton_atol = 1e-13``.

The next parameter for the Newton iteration is the maximum number of iterations it
is allowed to perform before the iteration is cancelled. This is controlled via ::

    newton_iter = 50

which defaults to ``newton_iter = 50``.

The parameter ``newton_damped``, which is set via ::

    newton_damped = True

is a boolean flag, indicating whether a damping strategy should be performed for the
Newton method, or whether the classical Newton-Raphson iteration shall be used. This
defaults to ``True``, but for some problems it might be beneficial (and faster) to not
use damping.

Next, we have the parameter ::

    newton_verbose = False

This is used to toggle the verbose output of the Newton method for the state system.
By default this is set to ``False`` so that there is not too much noise in the terminal.


The upcoming parameters are used to define the behavior of a Picard iteration, that
may be used if we have multiple variables.

.. note::

    For a detailed discussion of how to use the Picard iteration to solve a coupled
    state system, we refer to :ref:`demo_picard_iteration`. Note, that this demo
    is written for optimal control problems, but the definition of the state system
    can be transferred analogously to shape optimization problems, too.

First, we have a boolean flag, set via ::

    picard_iteration = False

which determines whether the Picard iteration is enabled or not. This defaults
to ``picard_iteration = False``, so that the Picard solver is disabled by default.
The following two parameters determine, analogously to above, the tolerances for the
Picard iteration ::

    picard_rtol = 1e-10
    picard_atol = 1e-12

The default values for these parameters are ``picard_rtol = 1e-10`` and
``picard_atol = 1e-12``. Moreover, note that the tolerances of the Newton solver are adjusted automatically in case
a Picard iteration is performedm, so that an inexact Picard iteration is used.

The maximum amout of iterations for the Picard iteration are set with ::

    picard_iter = 10

The default value for this is given by ``picard_iter = 50``.

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

There are three possible choices for this parameter for shape optimization problems, namely

- ``gd`` or ``gradient_descent`` : A gradient descent method

- ``cg``, ``conjugate_gradient``, ``ncg``, ``nonlinear_cg`` : Nonlinear CG methods

- ``lbfgs`` or ``bfgs`` : limited memory BFGS method.


Thereafter, we specify the tolerances for the optimization algorithm with the parameters ::

    rtol = 5e-3
    atol = 0.0

Again, ``rtol`` denotes the relative, and ``atol`` the absolute tolerance, and the
defaults for these parameters are given by ``rtol = 1e-3``, and ``atol = 0.0``.

The next parameter is used to control the maximum number of iterations performed by
the optimization algorithm. It is set via ::

    maximum_iterations = 50

and defaults to ``maximum_iterations = 100``.

Next up, we have the initial guess for the step size, which can be determined via ::

    initial_stepsize = 1.0

The default behavior is given by ``initial_stepsize = 1.0``.

The upcoming parameters are used for the Armijo rule ::

    epsilon_armijo = 1e-4
    beta_armijo = 2

They are used to verify that the condition

.. math:: J((I + t \mathcal{V})\Omega) \leq J(\Omega) + \varepsilon_{\text{Armijo}}\ t\ dJ(\Omega)[\mathcal{V}]

holds, and if this is not satisfied, the stepsize is updated via :math:`t = \frac{t}{\beta_{\text{Armijo}}}`.
As default values for these parameters we use ``epsilon_armijo = 1e-4`` as well
as ``beta_armijo = 2``.

The following parameter, ``soft_exit``, is a boolean flag which determines how
the optimization algorithm is terminated in case it does not converge. If ``soft_exit = True``, then an
error message is printed, but code after the :py:meth:`solve <cashocs.ShapeOptimizationProblem.solve>` call of the
optimization problem will still be executed. However, when ``soft_exit = False``, CASHOCS
raises an exception and terminates. This is set via ::

    soft_exit = False

and is set to ``False`` by default.


.. _config_shape_algolbfgs:

Section AlgoLBFGS
-----------------

Next, we discuss the parameters relevant for the limited memory BFGS method. For details
regarding this method, we refer to `Schulz, Siebenborn, and Welker, Efficient PDE Constrained Shape Optimization Based on Steklov-Poincaré-Type Metrics
<https://doi.org/10.1137/15M1029369>`_, where the methods are introduced.

The first parameter, ``bfgs_memory_size``, determines how large the storage of the BFGS method is. It is set via ::

    bfgs_memory_size = 3

Usually, a higher storage leads to a better Hessian approximation, and thus to faster
convergence. However, this also leads to an increased memory usage. Typically, values
below 5 already work very well. The default is ``bfgs_memory_size = 5``.

The other parameter for the BFGS method is ::

    use_bfgs_scaling = True

This determines, whether one should use a scaling of the initial Hessian approximation
(see `Nocedal and Wright, Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_).
This is usually very beneficial and should be kept enabled (which is the default).

.. _config_shape_algocg:

Section AlgoCG
--------------

The following parameters are used to define the behavior of the nonlinear conjugate
gradient methods for shape optimization. For more details on this, we refer to the
preprint `Blauth, Nonlinear Conjugate Gradient Methods for PDE Constrained Shape
Optimization Based on Steklov-Poincaré-Type Metrics <https://arxiv.org/abs/2007.12891>`_.

First, we define which nonlinear CG method is used by ::

    cg_method = DY

Available options are

- ``FR`` : The Fletcher-Reeves method

- ``PR`` : The Polak-Ribiere method

- ``HS`` : The Hestenes-Stiefel method

- ``DY`` : The Dai-Yuan method

- ``HZ`` : The Hager-Zhang method

The default value is ``cg_method = FR``. As for optimal control problems, the subsequent parameters are used to define the
restart behavior of the nonlinear CG methods. First, we have ::

    cg_periodic_restart = False

This boolean flag en- or disables that the NCG methods are restarted after a fixed
amount of iterations, which is specified via ::

    cg_periodic_its = 5

i.e., if ``cg_periodic_restart = True`` and ``cg_periodic_its = n``, then the NCG method
is restarted every ``n`` iterations. The default behavior is given by
``cg_periodic_restart = False`` and ``cg_periodic_its = 10``.

Alternatively, there also exists a relative restart criterion (see `Nocedal and Wright,
Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_), which can be enabled
via the boolean flag ``cg_relative_restart``, which is defined in the line ::

    cg_relative_restart = False

and the corresponding restart tolerance is set in ::

    cg_restart_tol = 0.5

Note, that ``cg_restart_tol`` should be in :math:`(0, 1)`. If two subsequent
gradients generated by the nonlinear CG method are not "sufficiently
orthogonal", the method is restarted with a gradient step. The default behavior
is given by ``cg_relative_restart = False`` and ``cg_restart_tol = 0.25``.

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
form is given as follows, in a general form, that is also implemented in CASHOCS

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
Moreover, we note that for the second Lamé parameter :math:`\mu`, CASHOCS implements
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

Moreover, let us take a look at the possible types of boundaries that can be used
with CASHOCS. In principle, there exist
two types: deformable and fixed boundaries. On fixed boundaries, we
impose homogeneous Dirichlet boundary conditions for the shape gradient, so that
these are not moved under the corresponding deformation. In CASHOCS, we define what boundaries
are fixed and deformable via their markers, which are either defined in the
corresponding python script, or in the GMSH file, if such a mesh is imported.

The config file for :ref:`demo_shape_poisson` defines the deformable boundaries
with the command ::

    shape_bdry_def = [1]

.. note::

    Remember, that in :ref:`demo_shape_poisson`, we defined ``boundaries`` with the commands ::

        boundary = CompiledSubDomain('on_boundary')
        boundaries = MeshFunction('size_t', mesh, dim=1)
        boundary.mark(boundaries, 1)

    Hence, we see that the marker ``1`` corresponds to the entire boundary, so that this
    is set to being deformable through the config.

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

The next parameter is specified via ::

    use_pull_back = True

This parameter is used to determine, whether the material derivative should
be computed for objects that are not state or adjoint variables. This is
enabled by default.

.. warning::

    This parameter should always be set to ``True``, otherwise the shape derivative might
    be wrong. Only disable it when you are sure what you are doing.

    Furthermore, note that the material derivative computation is only correct,
    as long as no differential operators act on objects that are not state or
    adjoint variables. However, this should only be a minor restriction and not
    relevant for almost all problems.

.. note::

    See :ref:`demo_inverse_tomography` for a case, where we use
    ``use_pull_back = False``.

The next parameters determine the coefficients of the bilinear form :math:`a`.
First, we have the first Lamé parameter :math:`\lambda`, which is set via ::

    lambda_lame = 1.428571428571429

The default value for this is ``lambda_lame = 0.0``.

Next, we specify the damping parameter :math:`\delta` with the line ::

    damping_factor = 0.2

The default for this is ``damping_factor = 0.0``.

.. note::

    As the default value for the damping factor is ``damping_factor = 0.0``, this
    should be set to a positive value in case the entire boundary of a problem
    is deformable. Otherwise, the Riesz identification problem for the shape
    gradient is not well-posed.

Finally, we define the values for :math:`\mu_\text{def}` and :math:`\mu_\text{fix}`
via ::

    mu_fix = 0.35714285714285715
    mu_def = 0.35714285714285715

The default behavior is given by ``mu_fix = 1.0`` and ``mu_def = 1.0``.

The parameter ``use_sqrt_mu`` is a boolean flag, which switches between using
:math:`\mu` and :math:`\sqrt{\mu}` as the stiffness for the linear elasticity
equations, as discussed above. This is set via ::

    use_sqrt_mu = False

and the default value is ``use_sqrt_mu = False``.

The next line in the config file is ::

    inhomogeneous = False

This determines, whether an inhomogeneous linear elasticity equation is used to
project the shape gradient. This scales the parameters :math:`\mu, \lambda` and
:math:`\delta` by :math:`\frac{1}{\text{vol}}`, where :math:`\text{vol}` is the
volume of the current element (during assembly). This means, that smaller elements
get a higher stiffness, so that the deformation takes place in the larger elements,
which can handle larger deformations without reducing their quality too much. For
more details on this approach, we refer to the preprint `Blauth, Leithäuser, and Pinnau,
Model Hierarchy for the Shape Optimization of a Microchannel Cooling System
<https://arxiv.org/abs/1911.06819>`_.



.. _config_shape_regularization:

Section Regularization
----------------------

In this section, the parameters for shape regularizations are specified. For a
detailed discussion of their usage, we refer to :ref:`demo_regularization`.

First, we have the parameters ``factor_volume`` and ``target_volume``. These are set
via the lines ::

    factor_volume = 0.0
    target_volume = 3.14

They are used to implement the (target) volume regularization term

.. math::

    \frac{\mu_\text{vol}}{2} \left( \int_{\Omega} 1 \text{ d}x - \text{vol}_\text{des} \right)^2

Here, :math:`\mu_\text{vol}` is specified via ``factor_volume``, and :math:`\text{vol}_\text{des}`
is the target volume, specified via ``target_volume``. The default behavior is
``factor_volume = 0.0`` and ``target_volume = 0.0``, so that we do not have
a volume regularization.

The next line, i.e., ::

    use_initial_volume = True

determines the boolean flag ``use_initial_volume``. If this is set to ``True``,
then not the value given in ``target_volume`` is used, but instead the
volume of the initial geometry is used for :math:`\text{vol}_\text{des}`.

For the next two types of regularization, namely the (target) surface and (target)
barycenter regularization, the syntax for specifying the parameters is completely
analogous. For the (target) surface regularization we have ::

    factor_surface = 0.0
    target_surface = 1.0

These parameter are used to implement the regularization term

.. math::

    \frac{\mu_\text{surf}}{2} \left( \int_{\Gamma} 1 \text{ d}s - \text{surf}_\text{des} \right)^2

Here, :math:`\mu_\text{surf}` is determined via ``factor_surface``, and
:math:`\text{surf}_\text{des}` is determined via ``target_surface``. The default
values are given by ``factor_surface = 0.0`` and ``target_surface = 0.0``.

As for the volume regularization, the parameter ::

    use_initial_surface = True

determines whether the target surface area is specified via ``target_surface``
or if the surface area of the initial geometry should be used instead. The default
behavior is given by ``use_initial_surface = False``.

Next, we have the curvature regularization, which is controlled by the parameter ::

    factor_curvature = 0.0

This is used to determine the size of :math:`\mu_\text{curv}` in the regularization
term

.. math::

    \frac{\mu_\text{curv}}{2} \int_{\Gamma} \kappa^2 \text{ d}s,

where :math:`\kappa` denotes the mean curvature. This regularization term can be
used to generate more smooth boundaries and to prevent kinks from occurring.

Finally, we have the (target) barycenter regularization. This is specified via
the parameters ::

    factor_barycenter = 0.0
    target_barycenter = [0.0, 0.0, 0.0]

and implements the term

.. math::

    \frac{\mu_\text{bary}}{2} \left\lvert \frac{1}{\text{vol}(\Omega)} \int_\Omega x \text{ d}x - \text{bary}_\text{des} \right\rvert^2

The default behavior is given by ``factor_barycenter = 0.0`` and ``target_barycenter = [0,0,0]``,
so that we do not have a barycenter regularization.

The flag ::

    use_initial_barycenter = True

again determines, whether :math:`\text{bary}_\text{des}` is determined via ``target_barycenter``
or if the barycenter of the initial geometry should be used instead. The default behavior
is given by ``use_initial_barycenter = False``.

.. hint::

    The object ``target_barycenter`` has to be a list. For 2D problems it is also
    sufficient, if the list only has two entries, for the :math:`x` and :math:`y`
    barycenters.

Finally, we have the parameter ``use_relative_scaling`` which is set in the line

    use_relative_scaling = False

This boolean flag does the following. For some regularization term :math:`J_\text{reg}(\Omega)` with corresponding
factor :math:`\mu` (as defined above), the default behavior is given by ``use_relative_scaling = False``
adds the term :math:`\mu J_\text{reg}(\Omega)` to the cost functional, so that the
factor specified in the configuration file is actually used as the factor for the regularization term.
In case ``use_relative_scaling = True``, the behavior is different, and the following term is
added to the cost functional: :math:`\frac{\mu}{\left\lvert J_\text{reg}(\Omega_0) \right\rvert} J_\text{reg}(\Omega)`,
where :math:`\Omega_0` is the initial guess for the geometry. In particular, this means
that the magnitude of the regularization term is equal to :math:`\mu` on the initial geometry.
This allows a detailed weighting of multiple regularization terms, which is particularly
useful in case the cost functional is also scaled (see :ref:`demo_scaling`).

.. _config_shape_mesh_quality:

Section MeshQuality
-------------------

This section details the parameters that influence the quality of the
computational mesh. First, we have the lines ::

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

Here, :math:`\alpha` corresponds to ``volume_change`` and :math:`\beta` corresponds
to ``angle_change``, and :math:`\mathcal{V}` is the deformation. The default behavior
is given by ``volume_change = inf`` and ``angle_change = inf``, so that no restrictions
are posed. Note, that, e.g., `Etling, Herzog, Loayza, Wachsmuth,
First and Second Order Shape Optimization Based on Restricted Mesh Deformations
<https://doi.org/10.1137/19M1241465>`_ use the values ``volume_change = 2.0`` and
``angle_change = 0.3``.

The next two parameters are given by ::

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
  to be deformed, has at least a quality of ``tol_lupper``, so that this quality
  might be reached again, if the step size is just decreased sufficiently often.
  This way, it is ensured that the state system is only solved when the mesh quality
  is larger than ``tol_lower``, so that the corresponding cost functional value is
  reasonable.

The default behavior is given by ``tol_lower = 0.0`` and ``tol_upper = 1e-15``,
so that there are basically no requirements on the mesh quality.

Finally, the upcoming two parameters specify how exactly the mesh quality is measured.
The first one is ::

    measure = condition_number

and determines one of the four mesh quality criteria, as defined in :py:class:`MeshQuality <cashocs.MeshQuality>`.
Available options are

- ``skewness``
- ``maximum_angle``
- ``radius_ratios``
- ``condition_number``

(see :py:class:`MeshQuality <cashocs.MeshQuality>` for a detailed description).
The default value is given by ``measure = skewness``.

Finally, the parameter ``type`` determines, whether the minimum quality over all
elements (``type = min``) or the average quality over all elements (``type = avg``)
shall be used. This is set via ::

    type = min

and defaults to ``type = min``.

.. _config_shape_output:

Section Output
--------------

In this section, the parameters for the output of the algorithm, either in the terminal
or as files, are specified. First, we have the parameter ``verbose``. This is used to toggle the output of the
optimization algorithm. It defaults to ``True`` and is controlled via ::

    verbose = True

The parameter ``save_results`` is a boolean flag, which determines whether a history
of the optimization algorithm, including cost functional value, gradient norm, accepted
step sizes, and mesh quality, shall be saved to a .json file. This defaults to ``True``,
and can be set with ::

    save_results = False

Moreover, we define the parameter ``save_txt`` ::
	
	save_txt = False

This saves the output of the optimization, which is usually shown in the terminal,
to a .txt file, which is human-readable.

The next line in the config file is ::

    save_pvd = False

Here, the parameter ``save_pvd`` is set. This is a boolean flag, which can be set to
``True`` to enable that CASHOCS generates .pvd files for the state variables for each iteration the optimization algorithm performs. These are great for visualizing the
steps done by the optimization algorithm, but also need some disc space, so that they are disabled by default.
Note, that for visualizing these files, you need `Paraview <https://www.paraview.org/>`_.

The next parameter, ``save_pvd_adjoint`` works analogously, and is given in the line ::

    save_pvd_adjoint = False

If this is set to True, CASHOCS generates .pvd files for the adjoint variables in each iteration of the optimization algorithm.
Its main purpose is for debugging.

The next parameter is given by ``save_pvd_gradient``, which is given in the line ::

    save_pvd_gradient = False

This boolean flag ensures that a paraview with the computed shape gradient is saved in ``result_dir/pvd``. The main purpose of this is for debugging.

Moreover, we also have the parameter ``save_mesh`` that is set via ::

    save_mesh = False

This is used to save the optimized geometry to a GMSH file. The default behavior
is given by ``save_mesh = False``. Note, that this is only
possible if the input mesh was already generated by GMSH, and specified in :ref:`the Mesh
section of the config file <config_shape_mesh>`. For any other meshes, the underlying mesh is also saved in
the .pvd files, so that you can at least always visualize the optimized geometry.

In the end, we also have, like for optimal control problems, a parameter that specifies
where the output is placed, again named ``result_dir``, which is given in the config file
in the line ::

    result_dir = ./

As before, this is either a relative or absolute path to the directory where the
results should be placed.


.. _config_shape_summary:

Summary
-------

Finally, an overview over all parameters and their default values can be found
in the following.


[Mesh]
******

.. list-table::
    :header-rows: 1

    * - Parameters
      - Default value
      - Remarks
    * - mesh_file
      -
      - Only needed for remeshing
    * - gmsh_file
      -
      - Only needed for remeshing
    * - geo_file
      -
      - Only needed for remeshing
    * - remesh
      - ``False``
      - if ``True``, remeshing is enabled; this feature is experimental, use with care
    * - show_gmsh_output
      - ``False``
      - if ``True``, shows the output of GMSH during remeshing in the console



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



[ShapeGradient]
***************

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - shape_bdry_def
      - ``[]``
      - list of indices for the deformable boundaries
    * - shape_bdry_fix
      - ``[]``
      - list of indices for the fixed boundaries
    * - shape_bdry_fix_x
      - ``[]``
      - list of indices for boundaries with fixed x values
    * - shape_bdry_fix_y
      - ``[]``
      - list of indices for boundaries with fixed y values
    * - shape_bdry_fix_z
      - ``[]``
      - list of indices for boundaries with fixed z values
    * - use_pull_back
      - ``True``
      - if ``False``, shape derivative might be wrong; no pull-back for the material derivative is performed;
        only use with caution
    * - lambda_lame
      - ``0.0``
      - value of the first Lamé parameter for the elasticity equations
    * - damping_factor
      - ``0.0``
      - value of the damping parameter for the elasticity equations
    * - mu_def
      - ``1.0``
      - value of the second Lamé parameter on the deformable boundaries
    * - mu_fix
      - ``1.0``
      - value of the second Lamé parameter on the fixed boundaries
    * - use_sqrt_mu
      - ``False``
      - if ``True``, uses the square root of the computed ``mu_lame``; might be good for 3D problems
    * - inhomogeneous
      - ``False``
      - if ``True``, uses inhomogeneous elasticity equations, weighted by the local mesh size


[Regularization]
****************

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - factor_volume
      - ``0.0``
      - value of the regularization parameter for volume regularization; needs to be non-negative
    * - target_volume
      - ``0.0``
      - prescribed volume for the volume regularization
    * - use_initial_volume
      - ``False``
      - if ``True`` uses the volume of the initial geometry as prescribed volume
    * - factor_surface
      - ``0.0``
      - value of the regularization parameter for surface regularization; needs to be non-negative
    * - target_surface
      - ``0.0``
      - prescribed surface for the surface regularization
    * - use_initial_surface
      - ``False``
      - if ``True`` uses the surface area of the initial geometry as prescribed surface
    * - factor_curvature
      - ``0.0``
      - value of the regularization parameter for curvature regularization; needs to be non-negative
    * - factor_barycenter
      - ``0.0``
      - value of the regularization parameter for barycenter regularization; needs to be non-negative
    * - target_barycenter
      - ``[0.0, 0.0, 0.0]``
      - prescribed barycenter for the barycenter regularization
    * - use_initial_barycenter
      - ``False``
      - if ``True`` uses the barycenter of the initial geometry as prescribed barycenter
    * - use_relative_scaling
      - ``False``
      - if ``True``, then the regularization terms are scaled in such a way, that
        their magnitude on the initial geometry is equal to the quantity specified in
        ``factor_volume``, ``factor_surface``, ``factor_curvature``, and ``factor_barycenter``.



[MeshQuality]
*************

.. list-table::
    :header-rows: 1

    * - Parameter
      - Default value
      - Remarks
    * - volume_change
      - ``inf``
      - determines by what factor the volume of a cell is allowed to change within a single deformation
    * - angle_change
      - ``inf``
      - determines how much the angles of a cell are allowed to change within a single deformation
    * - tol_lower
      - ``0.0``
      - if the mesh quality is lower than this tolerance, the state system is not solved
        for the Armijo rule, instead step size is decreased
    * - tol_upper
      - ``1e-15``
      - if the mesh quality is between ``tol_lower`` and ``tol_upper``, the state
        system will still be solved for the Armijo rule. If the accepted step yields a quality
        lower than this, algorithm is terminated (or remeshing is initiated)
    * - measure
      - ``skewness``
      - determines which quality measure is used
    * - type
      - ``min``
      - determines if minimal or average quality is considered




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
      - if ``True``, the history of the shape gradient over the optimization is saved in .pvd files
    * - save_mesh
      - ``False``
      - if ``True``, saves the mesh for the optimized geometry; only available for GMSH input
    * - result_dir
      - ``./``
      - path to the directory, where the output should be placed
