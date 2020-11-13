.. _demo_regularization:

Regularization for Shape Optimization Problems
==============================================

Problem Formulation
-------------------

In this demo, we investigate how we can use regularizations for shape optimization
problems in CASHOCS. For our model problem, we use one similar to the one in :ref:`demo_shape_poisson`,
but which has additional regularization terms, i.e.,

.. math::

    \min_\Omega J(u, \Omega) = &\int_\Omega u \text{ d}x +
    \alpha_\text{vol} \int_\Omega 1 \text{ d}x +
    \alpha_\text{surf} \int_\Gamma 1 \text{ d}s \\
    &+
    \frac{\mu_\text{vol}}{2} \left( \int_\Omega 1 \text{ d}x - \text{vol}_\text{des} \right)^2 +
    \frac{\mu_\text{surf}}{2} \left( \int_\Gamma 1 \text{ d}s - \text{surf}_\text{des} \right)^2 \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u &= f \quad &&\text{ in } \Omega,\\
    u &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


For the initial domain, we use the unit disc :math:`\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}`, and the right-hand side :math:`f` is given by

.. math:: f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1,

as in :ref:`demo_shape_poisson`.

Implementation
--------------

The complete python code can be found in the file :download:`demo_regularization.py </../../demos/documented/shape_optimization/regularization/demo_regularization.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/regularization/config.ini>`.


Initialization
**************

The initial code, including the defition of the PDE constraint, is identical to
:ref:`demo_shape_poisson`, and uses the following code ::

    from fenics import *
    import cashocs


    config = cashocs.load_config('./config.ini')

    meshlevel = 15
    degree = 1
    dim = 2
    mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
    dx = Measure('dx', mesh)
    boundary = CompiledSubDomain('on_boundary')
    boundaries = MeshFunction('size_t', mesh, dim=1)
    boundary.mark(boundaries, 1)
    ds = Measure('ds', mesh, subdomain_data=boundaries)

    V = FunctionSpace(mesh, 'CG', 1)
    u = Function(V)
    p = Function(V)

    x = SpatialCoordinate(mesh)
    f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    e = inner(grad(u), grad(p))*dx - f*p*dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

Cost functional and regularization
**********************************

The only difference to :ref:`demo_shape_poisson` comes now, in the definition
of the cost functional which includes the additional regularization terms.

.. note::

    CASHOCS cannot treat the last two regularization terms directly by a user
    implementation. Instead, these regularization terms can be realized by setting
    the appropriate parameters in the config files, see the :ref:`Section Regularization <config_shape_regularization>`.

The first three summands of the cost functional can then be defined as ::

    alpha_vol = 1e-1
    alpha_surf = 1e-1

    J = u*dx + Constant(alpha_vol)*dx + Constant(alpha_surf)*ds

The remaining two parts are specified via :download:`config.ini
</../../demos/documented/shape_optimization/regularization/config.ini>`, where
the following lines are relevant ::

    [Regularization]
    factor_volume = 1.0
    target_volume = 1.5
    use_initial_volume = False
    factor_surface = 1.0
    target_surface = 4.5
    use_initial_surface = False

This sets the factor :math:`\mu_\text{vol}` to ``1.0``, :math:`\text{vol}_\text{des}`
to ``1.5``, :math:`\mu_\text{surf}` to ``1.0``, and :math:`\text{surf}_\text{des}`
to ``4.5``. Note, that ``use_initial_volume`` and ``use_initial_surface``
have to be set to ``False``, otherwise the corresponding quantities of the initial
geometry would be used instead of the ones prescribed in the config file.
The resulting regularization terms are then treated by CASHOCS, but are, except
for these definitions in the config file, invisible for the user.

Finally, we solve the problem as in :ref:`demo_shape_poisson` with the lines ::

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve()

The results should look like this

.. image:: /../../demos/documented/shape_optimization/regularization/img_regularization.png
