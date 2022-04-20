.. _demo_pre_post_hooks:

Pre- and Post-Hooks for the optimization
========================================

Problem Formulation
-------------------

In this demo we show how one can use the flexibility of cashocs
to "inject" their own code into the optimization, which is carried
out before each solve of the state system (``pre_hook``) or after each
gradient computation (``post_hook``). To do so, we investigate the following
problem

.. math::

    &\min\; J(u, c) = \frac{1}{2} \int_\Omega \left\lvert u - u_d \right\rvert^2 \text{ d}x + \frac{\alpha}{2} \int_\Omega \left\lvert c \right\rvert^2 \text{ d}x \\
    &\text{ subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u + Re (u\cdot \nabla) u + \nabla p &= c \quad &&\text{ in } \Omega, \\
    \text{div}(u) &= 0 \quad &&\text{ in } \Omega,\\
    u &= u_\text{dir} \quad &&\text{ on } \Gamma^\text{dir},\\
    u &= 0 \quad &&\text{ on } \Gamma^\text{no slip},\\
    p &= 0 \quad &&\text{ at } x^\text{pres}.
    \end{alignedat} \right.


In particular, the setting for this demo is very similar to the one of
:ref:`demo_stokes`, but here we consider the nonlinear incompressible Navier-Stokes equations.

In the following, we will describe how to solve this problem
using cashocs, where we will use the ``pre_hook`` functionality to implement
a homotopy for solving the Navier-Stokes equations.

Implementation
--------------
The complete python code can be found in the file :download:`demo_constraints.py </../../demos/documented/optimal_control/pre_post_hooks/demo_pre_post_hooks.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/pre_post_hooks/config.ini>`.

Initialization
**************

The initial part of the code is nearly identical to the one of :ref:`demo_stokes`, but here we use the
Navier-Stokes equations instead of the linear Stokes system ::

    from fenics import *

    import cashocs

    config = cashocs.load_config("./config.ini")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(30)

    v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))
    U = VectorFunctionSpace(mesh, "CG", 1)

    up = Function(V)
    u, p = split(up)
    vq = Function(V)
    v, q = split(vq)
    c = Function(U)

    Re = Constant(1e2)

    e = (
        inner(grad(u), grad(v)) * dx
        + Constant(Re) * dot(grad(u) * u, v) * dx
        - p * div(v) * dx
        - q * div(u) * dx
        - inner(c, v) * dx
    )


    def pressure_point(x, on_boundary):
        return near(x[0], 0) and near(x[1], 0)


    no_slip_bcs = cashocs.create_dirichlet_bcs(
        V.sub(0), Constant((0, 0)), boundaries, [1, 2, 3]
    )
    lid_velocity = Expression(("4*x[0]*(1-x[0])", "0.0"), degree=2)
    bc_lid = DirichletBC(V.sub(0), lid_velocity, boundaries, 4)
    bc_pressure = DirichletBC(V.sub(1), Constant(0), pressure_point, method="pointwise")
    bcs = no_slip_bcs + [bc_lid, bc_pressure]

    alpha = 1e-5
    u_d = Expression(
        (
            "sqrt(pow(x[0], 2) + pow(x[1], 2))*cos(2*pi*x[1])",
            "-sqrt(pow(x[0], 2) + pow(x[1], 2))*sin(2*pi*x[0])",
        ),
        degree=2,
    )
    J = cashocs.IntegralFunctional(
        Constant(0.5) * inner(u - u_d, u - u_d) * dx
        + Constant(0.5 * alpha) * inner(c, c) * dx
    )

pre-hooks
*********

Note, that we have chosen a Reynolds number of ``Re = 1e2`` for this demo. 
In order to solve the Navier-Stokes equations for higher Reynolds numbers, it is 
often sensible to first solve the equations for a lower Reynolds number and then 
use this solution as initial guess for the original high Reynolds number problem.
We can use this procedure in cashocs with its ``pre_hook`` functionality. 
A ``pre_hook`` is a function without arguments, which gets called each time before solving the
state equation. In our case, the ``pre_hook`` should solve the Navier-Stokes equations for a lower
Reynolds number, so we define it as follows ::

    def pre_hook():
        print("Solving low Re Navier-Stokes equations for homotopy.")
        v, q = TestFunctions(V)
        e = (
            inner(grad(u), grad(v)) * dx
            + Constant(Re / 10.0) * dot(grad(u) * u, v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
            - inner(c, v) * dx
        )
        cashocs.newton_solve(e, up, bcs, verbose=False)
        
where we solve the Navier-Stokes equations with a lower Reynolds number of ``Re / 10.0``.
Later on, we inject this ``pre_hook`` into cashocs by using the 
:py:meth:`inject_pre_hook <cashocs.OptimalControlProblem.inject_pre_hook>` method.

Additionally, cashocs implements the functionality of also performing a pre-defined action
after each gradient computation, given by a so-called ``post_hook``. In our case, we just want 
to print a statement so that we can visualize what is happening. Therefore, we define our ``post_hook``
as ::

    def post_hook():
        print("Performing an action after computing the gradient.")

Next, before we can inject these two hooks, we first have to define the optimal control problem ::

    ocp = cashocs.OptimalControlProblem(e, bcs, J, up, c, vq, config)

Finally, we can inject both hooks via ::

    ocp.inject_pre_hook(pre_hook)
    ocp.inject_post_hook(post_hook)
    
.. note::

    We can also save one line and use the code ::
    
        ocp.inject_pre_post_hook(pre_hook, post_hook)
    
    which is equivalent to the above two lines.
    
And in the end, we solve the problem with ::

    ocp.solve()

The resulting optimized control and state look as follows

.. image:: /../../demos/documented/optimal_control/pre_post_hooks/img_pre_post_hooks.png
