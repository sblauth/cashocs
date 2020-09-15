.. _demo_shape_poisson:

Shape Optimization with a Poisson Problem
=========================================

Problem Formulation
-------------------

In this demo, we investigate the basics of CASHOCS for shape optimization problems.
As a model problem, we investigate the following one from
`Etling, Herzog, Loayza, Wachsmuth, First and Second Order Shape Optimization Based on Restricted Mesh Deformations <https://doi.org/10.1137/19M1241465>`_

.. math::

    &\min_\Omega J(u, \Omega) = \int_\Omega u \text{ d}x \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u &= f \quad &&\text{ in } \Omega,\\
    u &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


For the initial domain, we use the unit disc :math:`\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}`, and the right-hand side :math:`f` is given by

.. math:: f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.


Implementation
--------------

The complete python code can be found in the file :download:`demo_shape_poisson.py </../../demos/documented/shape_optimization/shape_poisson/demo_shape_poisson.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/shape_poisson/config.ini>`.


Initialization
**************

We start the problem by using a wildcard import for FEniCS, and by importing CASHOCS ::

    from fenics import *
    import cashocs

Similarly to the optimal control case, we also require config files for shape
optimization problems in CASHOCS. A detailed discussion of the config files
for shape optimization is given in :ref:`config_shape_optimization`.
We read the config file with the :py:func:`create_config <cashocs.create_config>` command ::

    config = cashocs.create_config('./config.ini')

Next, we have to define the mesh. As the above problem is posed on the unit disc
initially, we define this via FEniCS commands (CASHOCS only has rectangular meshes built
in). This is done via the following code ::

    meshlevel = 10
    degree = 1
    dim = 2
    mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)

Next up, we define the :py:class:`fenics.Measure` objects, which we need to define
the problem. For the volume measure, we can simply invoke ::

    dx = Measure('dx', mesh)

However, for the surface measure, we need to mark the boundary. This is required since
CASHOCS distinguishes between three types of boundaries: The deformable boundary, the
fixed boundary, and boundaries that can only be deformed perpendicular to a certain
coordinate axis (see :ref:`the relevant documentation of the config files <config_shape_shape_gradient>`). Here, we investigate the
case of a completely deformable boundary, which makes things rather
easy. We mark this boundary with the marker ``1`` with the following piece of code ::

    boundary = CompiledSubDomain('on_boundary')
    boundaries = MeshFunction('size_t', mesh, dim=1)
    boundary.mark(boundaries, 1)
    ds = Measure('ds', mesh, subdomain_data=boundaries)

.. note::

    In :download:`config.ini </../../demos/documented/shape_optimization/shape_poisson/config.ini>`,
    in the section :ref:`ShapeGradient <config_shape_shape_gradient>`, there is
    the line ::

        shape_bdry_def = [1]

    which specifies that the boundary marked with 1 is deformable. For our
    example this is exactly what we want, as this means that the entire boundary
    is variable, due to the previous commands. For a detailed documentation we
    refer to :ref:`the corresponding documentation of the ShapeGradient section
    <config_shape_shape_gradient>`.

Note, that all of the alternative ways of marking subdomains or boundaries with
numbers, as explained in `Langtangen and Logg, Solving PDEs in Python
<https://doi.org/10.1007/978-3-319-52462-7>`_ also work here. If it is valid for FEniCS, it is also for
CASHOCS.

After having defined the initial geometry, we define a :py:class:`fenics.FunctionSpace` consisting of
piecewise linear Lagrange elements via ::

    V = FunctionSpace(mesh, 'CG', 1)
    u = Function(V)
    p = Function(V)

This also defines our state variable :math:`u` as ``u``, and the adjoint state :math:`p` is given by
``p``.

.. note::

    As remarked in :ref:`demo_poisson`, in
    classical FEniCS syntax we would use a :py:class:`fenics.TrialFunction` for ``u``
    and a :py:class:`fenics.TestFunction` for ``p``. However, for CASHOCS this must not
    be the case. Instead, the state and adjoint variables have to be :py:class:`fenics.Function` objects.

The right-hand side of the PDE constraint is then defined as ::

    x = SpatialCoordinate(mesh)
    f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

which allows us to define the weak form of the state equation via ::

    e = inner(grad(u), grad(p))*dx - f*p*dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

The optimization problem and its solution
*****************************************

We are now almost done, the only thing left to do is to define the cost functional ::

    J = u*dx

and the shape optimization problem ::

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)

This can then be solved in complete analogy to :ref:`demo_poisson` with
the :py:meth:`sop.solve() <cashocs.ShapeOptimizationProblem.solve>` command ::

    sop.solve()

The result of the optimization looks like this


.. image:: /../../demos/documented/shape_optimization/shape_poisson/img_shape_poisson.png

.. note::

    As in :ref:`demo_poisson` we can specify some keyword
    arguments for the :py:meth:`solve <cashocs.ShapeOptimizationProblem.solve>` command.
    If none are given, then the settings from the config file are used, but if
    some are given, they override the parameters specified
    in the config file. In particular, these arguments are

      - ``algorithm`` : Specifies which solution algorithm shall be used.
      - ``rtol`` : The relative tolerance for the optimization algorithm.
      - ``atol`` : The absolute tolerance for the optimization algorithm.
      - ``max_iter`` : The maximum amount of iterations that can be carried out.

    The possible choices for these parameters are discussed in detail in
    :ref:`config_shape_optimization_routine` and the documentation of the :py:func:`solve <cashocs.ShapeOptimizationProblem.solve>`
    method.
