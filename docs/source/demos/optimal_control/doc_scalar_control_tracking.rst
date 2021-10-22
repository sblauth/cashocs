.. _demo_scalar_control_tracking:

Tracking of the Cost Functional for Optimal Control Problems
============================================================


Problem Formulation
-------------------

In this demo we investigate CASHOCS functionality of tracking scalar-type
terms such as cost functional values and other quanitites, which typically
arise after integration. For this, we investigate the problem

.. math::

    &\min\; J(y,u) = \frac{1}{2} \left\lvert \int_{\Omega} y^2
    \text{ d}x - C_{des} \right\rvert^2 \\
    &\text{ subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta y &= u \quad &&\text{ in } \Omega,\\
    y &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


For this example, we do not consider control constraints,
but search for an optimal control u in the entire space :math:`L^2(\Omega)`,
for the sake of simplicitiy. For the domain under consideration, we use the unit square
:math:`\Omega = (0, 1)^2`, since this is built into CASHOCS.

In the following, we will describe how to solve this problem
using CASHOCS. Moreover,
we also detail alternative / equivalent FEniCS code which could
be used to define the problem instead.

Implementation
--------------
The complete python code can be found in the file :download:`demo_scalar_control_tracking.py </../../demos/documented/optimal_control/scalar_control_tracking/demo_scalar_control_tracking.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/scalar_control_tracking/config.ini>`.


The state problem
*****************

The difference to :ref:`demo_poisson` is that the cost functional does now track the
value of the :math:`L^2` norm of :math:`y` against a desired value of :math:`C_{des}`,
and not the state :math:`y` itself. Other than that, the corresponding PDE constraint
and its setup are completely analogous to :ref:`demo_poisson` ::

    from fenics import *

    import cashocs



    cashocs.set_log_level(cashocs.LogLevel.INFO)
    config = cashocs.load_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)
    u.vector()[:] = 1.0

    e = inner(grad(y), grad(p))*dx - u*p*dx

    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])


Definition of the scalar tracking type cost functional
******************************************************

Next, we define the cost functional. To do this in CASHOCS, we first have to set
the usual cost functional to :math:`0` by writing the line ::

    J = Constant(0)*dx

This ensures that only the other terms will be active

.. note::

    In case ``J`` is not defined as ``Constant(0)*dx`` but, e.g., like in
    :ref:`demo_poisson`, the terms will be added on top of each other.

To define the desired tracking type functional, note that CASHOCS implements the
functional for the following kind of cost functionals

.. math::

    \begin{aligned}
        J_vol(y,u) &= \frac{1}{2} \left\lvert \int_\Omega f(y,u) \text{ d}x - C_{des} \right\rvert^2 \\
        J_surf(y,u) &= \frac{1}{2} \left\lvert \int_\Gamma g(y,u) \text{ d}s - D_{des} \right\rvert^2
    \end{aligned}

Of course, also integrals over interior boundaries could be considered, or integrals
over only a subset of :math:`\Omega` or :math:`\Gamma`. To uniquely define these
cost functionals, we only need to define the integrands, i.e.,

.. math::

    f(y,u) \quad \text{and} \quad g(y,u)

as well as the goals of the tracking type functionals, i.e.,

.. math::

    C_{des} \quad \text{and} \quad D_{des}.

We do this by defining a python dictionary, which includes these terms with the
keywords ``'integrand'`` and ``'tracking_goal'``. For our model problem, the integrand
is given by :math:`y^2 \text{ d}x`, which is defined in FEniCS via the line ::

    integrand = y*y*dx

For the desired value of the (squared) norm of :math:`y` we use the value :math:`1.0`,
i.e., we define ::

    tracking_goal = 1.0

This is then put into a dictionary as follows ::

    J_tracking = {'integrand' : integrand, 'tracking_goal' : tracking_goal}

.. note::

    We could also prescribe a list of multiple dicts of this type. In this case,
    each of the corresponding tracking type terms will be added up.

.. hint::

    For the scaling possibilities, which are described in detail in :ref:`demo_scaling`,
    we use the following convention: All desired weights are defined in the list
    ``desired_weights``. In case we have :math:`n` cost functionals defined in
    ``cost_functional_form`` (here, :math:`n \geq 1`), and :math:`m` additional cost functionals
    defined as ``scalar_tracking_forms`` (:math:`m \geq 0`), then the first :math:`n`
    entries of ``desired_weights`` correspond to the cost functionals given in
    ``cost_functional_form``, and the last :math:`m` entries correspond to the
    cost functionals defined in ``scalar_tracking_forms``.


Finally, we can set up our new optimization problem as we already know, but we
now use the keyword argument ::

    scalar_tracking_forms = J_tracking

to specify the correct cost functional for our problem.
As usual, this is then solved with the :py:meth:`solve <cashocs.OptimalControlProblem.solve>`
method of the optimization problem.

Finally, we visualize the results using matplotlib and the following code ::

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    fig = plot(u)
    plt.colorbar(fig, fraction=0.046, pad=0.04)
    plt.title('Control variable u')

    plt.subplot(1,2,2)
    fig = plot(y)
    plt.colorbar(fig, fraction=0.046, pad=0.04)
    plt.title('State variable y')

    plt.tight_layout()

The output should look like this

.. image:: /../../demos/documented/optimal_control/scalar_control_tracking/img_scalar_control_tracking.png
