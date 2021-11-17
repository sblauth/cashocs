.. _demo_state_constraints:

Optimal Control with State Constraints
======================================

Problem Formulation
-------------------

In this demo we investigate how state constraints can be handled in CASHOCS. Thanks to
the high level interface for solving (control-constrained) optimal control problems,
the state constrained case can be treated (approximately) using a Moreau-Yosida
regularization, which we show in the following. As model problem, we consider the
following one

.. math::

    &\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{ d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{ d}x \\
    &\text{ subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta y &= u \quad &&\text{ in } \Omega,\\
    y &= 0 \quad &&\text{ on } \Gamma, \\
    y &\leq \bar{y} \quad &&\text{ in } \Omega,
    \end{alignedat} \right.


see, e.g., `Hinze, Pinnau, Ulbrich, and Ulbrich, Optimization with PDE constraints <https://doi.org/10.1007/978-1-4020-8839-1>`_.

Moreau-Yosida regularization
****************************

Instead of solving this problem directly, the Moreau-Yosida regularization instead solves
a sequence of problems without state constraints which are of the form

.. math:: \min J_\gamma(y, u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{ d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{ d}x + \frac{1}{2\gamma} \int_\Omega \lvert \max\left( 0, \hat{\mu} + \gamma (y - \bar{y}) \right) \rvert^2 \text{ d}x

for :math:`\gamma \to +\infty`. We employ a simple homotopy method, and solve the problem for one value of :math:`\gamma`, and then use this solution as initial guess for the next
higher value of :math:`\gamma`. As initial guess we use the solution of the unconstrained
problem. For a detailed discussion of the Moreau-Yosida regularization, we refer the
reader to, e.g., `Hinze, Pinnau, Ulbrich, and Ulbrich, Optimization with PDE constraints
<https://doi.org/10.1007/978-1-4020-8839-1>`_.


Implementation
--------------

The complete python code can be found in the file :download:`demo_state_constraints.py </../../demos/documented/optimal_control/state_constraints/demo_state_constraints.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/state_constraints/config.ini>`.

The initial guess for the homotopy
**********************************

As mentioned earlier, we first solve the unconstrained problem to get an initial
guess for the homotopy method. This is done in complete analogy to :ref:`demo_poisson` ::

    from fenics import *
    import cashocs
    import numpy as np
    from ufl import Max


    config = cashocs.load_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

    e = inner(grad(y), grad(p))*dx - u*p*dx
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    y_d = Expression('sin(2*pi*x[0]*x[1])', degree=1)
    alpha = 1e-3
    J_init = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx
    ocp_init = cashocs.OptimalControlProblem(e, bcs, J_init, y, u, p, config)
    ocp_init.solve()


.. note::

    Cashocs automatically updates the user input during the runtime of the optimization
    algorithm. Hence, after the :py:meth:`ocp_init.solve() <cashocs.OptimalControlProblem.solve>`
    command has returned, the solution is already stored in ``u``.

The regularized problems
************************

For the homotopy method with the Moreau-Yosida regularization, we first define the upper
bound for the state :math:`\bar{y}` and select a sequence of values for :math:`\gamma` via ::

    y_bar = 1e-1
    gammas = [pow(10, i) for i in np.arange(1, 9, 3)]

Solving the regularized problems is then as simple as writing a ``for`` loop ::

    for gamma in gammas:

        J = J_init + cashocs.moreau_yosida_regularization(
            y, gamma, dx, upper_treshold=y_bar
        )

    	ocp_gamma = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    	ocp_gamma.solve()

Here, we use a ``for`` loop, define the new cost functional (with the new value of :math:`\gamma`),
set up the optimal control problem and solve it, as previously.

.. hint::

    Note, that we could have also defined ``y_bar`` as a :py:class:`fenics.Function`
    or :py:class:`fenics.Expression`, and
    the method would have worked exactly the same, the corresponding object just has to
    be a valid input for an UFL form.

.. note::
    We could have also defined the Moreau-Yosida regularization of the inequality constraint
    directly, with the following code ::

    	J = (
    		J_init
    		+ Constant(1 / (2 * gamma)) * pow(Max(0, Constant(gamma) * (y - y_bar)), 2) * dx
    	)

    However, this is directly implemented in :py:func:`cashocs.moreau_yosida_regularization`, which is why we use this function in the demo.

Validation of the method
************************

Finally, we perform a post processing to see whether the state constraint is
(approximately) satisfied. Therefore, we compute the maximum value of ``y``,
and compute the relative error between this and ``y_bar`` ::

    y_max = np.max(y.vector()[:])
    error = abs(y_max - y_bar) / abs(y_bar) * 100
    print('Maximum value of y: ' + str(y_max))
    print('Relative error between y_max and y_bar: ' + str(error) + ' %')

As the error is about 0.01 %, we observe that the regularization indeed works
as expected, and this tolerance is sufficiently low for practical applications.

The visualization of the solution looks as follows

.. image:: /../../demos/documented/optimal_control/state_constraints/img_state_constraints.png
