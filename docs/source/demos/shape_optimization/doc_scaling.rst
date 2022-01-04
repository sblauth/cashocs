.. _demo_scaling:

Scaling of the Cost Functional
==============================

Problem Formulation
-------------------

In this demo, we take a look at how cashocs can be used to scale cost functionals,
which is particularly useful in case one uses multiple terms to define the cost functional
and wants to weight them appropriately, e.g., if there are multiple competing objectives.
This demo investigates this problem by considering a slightly modified version of
the model shape optimization problem from :ref:`demo_shape_poisson`, i.e.,

.. math::

    &\min_\Omega J(u, \Omega) = \alpha \int_\Omega u \text{ d}x + \beta \int_\Omega 1 \text{ d}x \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u &= f \quad &&\text{ in } \Omega,\\
    u &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


For the initial domain, we use the unit disc :math:`\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}`, and the right-hand side :math:`f` is given by

.. math:: f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.

Our goal is to choose the parameters :math:`\alpha` and :math:`\beta` in such a way
that the magnitude of the second term is twice as big as the value of the first term for
the initial geometry.

Implementation
--------------

The complete python code can be found in the file :download:`demo_scaling.py </../../demos/documented/shape_optimization/scaling/demo_scaling.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/scaling/config.ini>`.


Initialization
**************

The definition of the PDE constraint is completely identical to the one described in
:ref:`demo_shape_poisson`, which we recall here for the sake of completeness ::

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


Definition of the Cost Functional
*********************************

Our goal is to choose :math:`\alpha` and :math:`\beta` in such a way that

.. math::
    \begin{aligned}
        \left\lvert \alpha \int_{\Omega_0} u \text{ d}x \right\rvert &= C_1,\\
        \left\lvert \beta \int_{\Omega_0} 1 \text{ d}x \right\rvert &= C_2,
    \end{aligned}

where :math:`\Omega_0` is the intial geometry. Here, we choose the value :math:`C_1 = 1` and :math:`C_2 = 2` without loss of generality.
This would then achieve our goal of having the second term (a volume regularization)
being twice as large as the first term in magnitude.

To implement this in cashocs, we do not specify a single UFL form for the cost functional
as in all previous demos, instead we supply a list of UFL forms into which we
put every single term of the cost functional. These are then scaled automatically
by cashocs and then added to obtain the actual cost functional.

Hence, let us first define the individual terms of the cost functional we consider,
and place them into a list ::

    J_1 = u*dx
    J_2 = Constant(1)*dx
    J_list = [J_1, J_2]

Afterwards, we have to create a second list which includes the values that the
terms of ``J_list`` should have on the initial geometry ::

    desired_weights = [1, 2]

Finally, these are supplied to the shape optimization problem as follows: ``J_list``
is passed instead of the usual ``cost_functional_form`` parameter, and ``desired_weights``
enters the optimization problem as keyword argument of the same name, i.e., ::

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J_list, u, p, boundaries, config, desired_weights=desired_weights)
    sop.solve()

.. note::

    Since the first term of the cost functional, i.e., :math:`\int_\Omega u \text{ d}x`,
    is negative, the intial function value for our choice of scaling is :math:`-1 + 2 = 1`.

.. note::

    If the keyword argument ``desired_weights`` is not given or ``None``, this is
    equivalent to not using the scaling described above, i.e., one could also just pass
    ``J = J_1 + J_2`` as ``cost_functional_form``.

.. note::

    If a cost functional is close to zero for the initial domain, the scaling is
    disabled for this term, and instead the respective term is just multiplied
    by the corresponding factor in ``desired_weights``. cashocs issues an info message
    in this case.

.. note::

    The scaling of the cost functional works completely analogous for optimal control problems:
    There, one also has to supply a list of the individual terms of the cost functional
    and use the keyword argument ``desired_weights`` in order to define and supply the
    desired magnitude of the terms for the initial iteration.

The result of the optimization looks like this

.. image:: /../../demos/documented/shape_optimization/scaling/img_scaling.png
