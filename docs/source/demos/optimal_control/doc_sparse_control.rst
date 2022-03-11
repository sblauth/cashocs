.. _demo_sparse_control:

Sparse Control
==============

In this demo, we investigate a possibility for obtaining sparse optimal controls.
To do so, we use a sparsity promoting :math:`L^1` regularization. Hence, our model problem
for this demo is given by

.. math::

    &\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{ d}x + \frac{\alpha}{2} \int_{\Omega} \lvert u \rvert \text{ d}x \\
    &\text{ subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta y &= u \quad &&\text{ in } \Omega,\\
    y &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


This is basically the same problem as in :ref:`demo_poisson`, but the regularization is now not the :math:`L^2` norm squared, but just the :math:`L^1` norm.

Implementation
--------------

The complete python code can be found in the file :download:`demo_sparse_control.py </../../demos/documented/optimal_control/sparse_control/demo_sparse_control.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/sparse_control/config.ini>`.



The implementation of this problem is completely analogous to the one of :ref:`demo_poisson`,
the only difference is the definition of the cost functional. We state the entire code
in the following. The only difference between this implementation and
the one of :ref:`demo_poisson` is in line 18, and is highlighted here.

.. code-block:: python
    :linenos:
    :emphasize-lines: 18

    from fenics import *

    import cashocs

    config = cashocs.load_config("config.ini")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
    V = FunctionSpace(mesh, "CG", 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

    e = inner(grad(y), grad(p)) * dx - u * p * dx
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
    alpha = 1e-4
    J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * abs(u) * dx

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    ocp.solve()

Note, that for the regularization term we now do not use ``Constant(0.5*alpha)*u*u*dx``,
which corresponds to the :math:`L^2(\Omega)` norm squared, but rather ::

    Constant(0.5 * alpha) * abs(u) * dx

which corresponds to the :math:`L^1(\Omega)` norm. Other than that, the code is identical.
The visualization of the code also shows, that we have indeed a sparse control

.. image:: /../../demos/documented/optimal_control/sparse_control/img_sparse_control.png

.. note::
    The oscillations in between the peaks for the control variable ``u`` are just numerical noise, which comes
    from the discretization error.
