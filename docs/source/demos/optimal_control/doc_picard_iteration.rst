.. _demo_picard_iteration:

Coupled Problems - Picard Iteration
===================================

Problem Formulation
-------------------

In this demo we show how CASHOCS can be used with a coupled PDE constraint.
For this, we consider a iterative approach, whereas we investigated
a monolithic approach in :ref:`demo_monolithic_problems`.

As model example, we consider the
following problem

.. math::

    &\min\; J((y,z),(u,v)) = \frac{1}{2} \int_\Omega \left( y - y_d \right)^2 \text{ d}x + \frac{1}{2} \int_\Omega \left( z - z_d \right)^2 \text{ d}x + \frac{\alpha}{2} \int_\Omega u^2 \text{ d}x + \frac{\beta}{2} \int_\Omega v^2 \text{ d}x \\
    &\text{ subject to }\quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta y + z &= u \quad &&\text{ in } \Omega, \\
    y &= 0 \quad &&\text{ on } \Gamma,\\
    -\Delta z + y &= v \quad &&\text{ in } \Omega,\\
    z &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.

Again, the system is two-way coupled. To solve it, we now employ a Picard iteration. Therefore,
the two PDEs are solved subsequently, where the variables are frozen in between: At the beginning
the first PDE is solved for :math:`y`, with :math:`z` being fixed. Afterwards, the second PDE is solved for :math:`z`
with :math:`y` fixed. This is then repeated
until convergence is reached.

.. note::

    There is, however, no a-priori guarantee that the Picard iteration converges
    for a particular problem, unless a careful analysis is carried out by the user.
    Still, it is an important tool, which also often works well in practice.

Implementation
--------------

The complete python code can be found in the file :download:`demo_picard_iteration.py </../../demos/documented/optimal_control/picard_iteration/demo_picard_iteration.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/picard_iteration/config.ini>`.

Initialization
**************

The setup is as in :ref:`demo_poisson` ::

    from fenics import *
    import cashocs


    config = cashocs.load_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
    V = FunctionSpace(mesh, 'CG', 1)

However, compared to the previous examples, there is a major change in the config file. As we want to use
the Picard iteration as solver for the state PDEs, we now specify ::

    picard_iteration = True

see :download:`config.ini </../../demos/documented/optimal_control/picard_iteration/config.ini>`.


Definition of the state system
******************************

The definition of the state system follows the same ideas as introduced in
:ref:`demo_multiple_variables`: We define both state equations through their components,
and then gather them in lists, which are passed to the :py:class:`OptimalControlProblem <cashocs.OptimalControlProblem>`.
The state and adjoint variables are defined via ::

    y = Function(V)
    p = Function(V)
    z = Function(V)
    q = Function(V)

The control variables are defined as ::

    u = Function(V)
    v = Function(V)

Next, we define the state system, using the weak forms from
:ref:`demo_monolithic_problems` ::

    e_y = inner(grad(y), grad(p))*dx + z*p*dx - u*p*dx
    bcs_y = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    e_z = inner(grad(z), grad(q))*dx + y*q*dx - v*q*dx
    bcs_z = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

Finally, we use the same procedure as in :ref:`demo_multiple_variables`, and
put everything into (ordered) lists

    states = [y, z]
    adjoints = [p, q]
    controls = [u, v]

    e = [e_y, e_z]
    bcs = [bcs_y, bcs_z]


Definition of the optimization problem
**************************************

The cost functional is defined as in :ref:`demo_monolithic_problems`, the only
difference is that ``y`` and ``z`` now are :py:class:`fenics.Function` objects, whereas they
were generated with the :py:func:`fenics.split` command previously ::

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
    alpha = 1e-6
    beta = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
        + Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

Finally, we set up the optimization problem and solve it ::

    optimization_problem = cashocs.OptimalControlProblem(e, bcs, J, states, controls, adjoints, config)
    optimization_problem.solve()

The result should look like this

.. image:: /../../demos/documented/optimal_control/picard_iteration/img_picard_iteration.png

.. note::

    Comparing the output (especially in the early iterations) between the monlithic and Picard apporach
    we observe that both methods yield essentially the same results (up to machine precision). This validates
    the Picard approach.

    However, one should note that for this example, the Picard approach takes significantly longer to
    compute the optimizer. This is due to the fact that the individual PDEs have to be solved several
    times, whereas in the monolithic approach the state system is (slightly) larger, but has to be solved
    less often. However, the monolithic approach needs significantly more memory, so that the Picard
    iteration becomes feasible for very large problems. Further, the convergence properties of the
    Picard iteration are better, so that it may converge even when the monolithic approach fails.
