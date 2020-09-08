.. _demo_dirichlet_control:

Dirichlet Boundary Control
==========================

Problem Formulation
-------------------

In this demo, we investigate how Dirichlet boundary control is possible with
CASHOCS. To do this, we have to employ the so-called Nitsche method, which we
briefly recall in the following. Our model problem for this example is given by

.. math::

    &\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{d}x \\
    &\text{ subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta y &= 0 \quad &&\text{ in } \Omega,\\
    y &= u \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


In contrast to our previous problems, the control now enters the problem as a
Dirichlet boundary condition. However, we cannot apply these via a :py:class:`fenics.DirichletBC`,
because for CASHOCS to work properly, the controls,
states, and adjoints are only allowed to appear in UFL forms. Nitsche's Method
circumvents this problem by imposing the boundary conditions in the weak form
directly. Let us first briefly recall this method.

Nitsche's method
****************

Consider the Laplace problem

.. math::
    \begin{alignedat}{2}
    -\Delta y &= 0 \quad &&\text{ in } \Omega,\\
    y &= u \quad &&\text{ on } \Gamma.
    \end{alignedat}

We can derive a weak form for this equation in :math:`H^1(\Omega)`
(not :math:`H^1_0(\Omega)`) by multiplying the equation by a test function
:math:`p \in H^1(\Omega)` and applying the divergence theorem

.. math:: \int_\Omega - \Delta y p \text{d}x = \int_\Omega \nabla y \cdot \nabla p \text{d}x - \int_\Gamma (\nabla y \cdot n) p \text{d}s.

This weak form is the starting point for Nitsche's method. First of all, observe that
this weak form is not symmetric anymore. To restore symmetry of the problem, we can
use the Dirichlet boundary condition and "add a zero" by adding :math:`\int_\Gamma \nabla p \cdot n (y - u) \text{d}s`. This gives the weak form

.. math:: \int_\Omega \nabla y \cdot \nabla p \text{d}x - \int_\Gamma (\nabla y \cdot n) p \text{d}s - \int_\Gamma (\nabla p \cdot n) y \text{d}s = \int_\Gamma (\nabla p \cdot n) u \text{d}s.

However, one can show that this weak form is not coercive. Hence, Nitsche's method
adds another zero to this weak form, namely :math:`\int_\Gamma \eta (y - u) p \text{d}s`,
which yields the coercivity of the problem if :math:`\eta` is sufficiently large. Hence,
we consider the following weak form

.. math:: \int_\Omega \nabla y \cdot \nabla p \text{d}x - \int_\Gamma (\nabla y \cdot n) p \text{d}s - \int_\Gamma (\nabla p \cdot n) y \text{d}s + \eta \int_\Gamma y p \text{d}s = \int_\Gamma (\nabla p \cdot n) u \text{d}s + \eta \int_\Gamma u p \text{d}s,

and this is the form we implement for this problem.

For a detailed introduction to Nitsche's method, we refer to
`Assous and Michaeli, A numerical method for handling boundary and
transmission conditions in some linear partial differential equations
<https://doi.org/10.1016/j.procs.2012.04.045>`_.


Implementation
--------------

The complete python code can be found in the file :download:`demo_dirichlet_control.py </../../demos/documented/optimal_control/dirichlet_control/demo_dirichlet_control.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/dirichlet_control/config.ini>`.

Initialization
**************

The beginning of the program is exactly the same as for :ref:`demo_poisson` ::

    from fenics import *
    import cashocs
    import numpy as np


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    n = FacetNormal(mesh)
    h = MaxCellEdgeLength(mesh)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)


Then, we define the Dirichlet boundary conditions. As we use Nitsche's method, there
are none, which we define by an empty list ::

    bcs = []


Definition of the PDE and optimization problem via Nitsche's method
*******************************************************************

Afterwards, we implement the weak form using Nitsche's method, as described above, which
is given by the code segment ::

    eta = Constant(1e4)
    e = inner(grad(y), grad(p))*dx - inner(grad(y), n)*p*ds - inner(grad(p), n)*(y - u)*ds + eta/h*(y - u)*p*ds - Constant(1)*p*dx

Finally, we can define the optimization problem similarly to :ref:`demo_neumann_control` ::

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    alpha = 1e-4
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*ds

As for :ref:`demo_neumann_control`, we have to define a scalar product on
:math:`L^2(\Gamma)` to get meaningful results (as the control is only defined on the boundary),
which we do with ::

    scalar_product = TrialFunction(V)*TestFunction(V)*ds

Solution of the optimization problem
************************************

The optimal control problem is solved with the usual syntax ::

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config, riesz_scalar_products=scalar_product)
    ocp.solve()

The result should look like this

.. image:: /../../demos/documented/optimal_control/dirichlet_control/img_dirichlet_control.png

.. note::

    In the end, we validate whether the boundary conditions are applied correctly
    using this approach. Therefore, we first compute the indices of all DOF's
    that lie on the boundary via ::

        bcs = cashocs.create_bcs_list(V, 1, boundaries, [1,2,3,4])
        bdry_idx = Function(V)
        [bc.apply(bdry_idx.vector()) for bc in bcs]
        mask = np.where(bdry_idx.vector()[:] == 1)[0]

    Then, we restrict both ``y`` and ``u`` to the boundary by ::

        y_bdry = Function(V)
        u_bdry = Function(V)
        y_bdry.vector()[mask] = y.vector()[mask]
        u_bdry.vector()[mask] = u.vector()[mask]

    Finally, we compute the relative errors in the :math:`L^\infty(\Gamma)` and
    :math:`L^2(\Gamma)` norms and print the result ::

        error_inf = np.max(np.abs(y_bdry.vector()[:] - u_bdry.vector()[:])) / np.max(np.abs(u_bdry.vector()[:])) * 100
        error_l2 = np.sqrt(assemble((y - u)*(y - u)*ds)) / np.sqrt(assemble(u*u*ds)) * 100

        print('Error regarding the (weak) imposition of the boundary values')
        print('Error L^\infty: ' + format(error_inf, '.3e') + ' %')
        print('Error L^2: ' + format(error_l2, '.3e') + ' %')

    We see, that with ``eta = 1e4`` we get a relative error of under 5e-3 %, which is
    more than sufficient for any application.