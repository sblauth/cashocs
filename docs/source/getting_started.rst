Getting Started
===============

As newcomer to CASHOCS, we recommend the paper `Blauth, cashocs: A Computational, Adjoint-Based
Shape Optimization and Optimal Control Software <https://doi.org/10.1016/j.softx.2020.100646>`_, which
gives an overview over CASHOCS and its capabilities. Moreover, for a
comprehensive description of CASHOCS, we refer to its :ref:`tutorial <tutorial_index>`.

In the following, we briefly showcase how CASHOCS can be used to solve a
distributed optimal control problem with a Poisson equation as PDE constaint.

Since CASHOCS is based on FEniCS, most of the user input consists of definining
the objects (such as the state system and cost functional) via UFL forms. If one
has a functioning code for the forward problem and the evaluation of the cost
functional, the necessary modifications to optimize the problem in CASHOCS
are minimal. Consider, e.g., the following optimization problem

.. math::

    &\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2
    \text{ d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{ d}x \\
    &\text{ subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta y &= u \quad &&\text{ in } \Omega,\\
    y &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.

Note, that the problem is treated in detail in the :ref:`corresponding CASHOCS tutorial <demo_poisson>`.

For our purposes, we assume that a mesh for this problem is defined and that a
suitable function space is chosen. This can, e.g., be achieved via ::

    from fenics import *
    import cashocs

    config = cashocs.load_config('path_to_config')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

The config object, which is created from a .ini file, is used to determine the
parameters for the optimization algorithms. This is where the user can finely
tune the behavior of the algorithms.

To define the state problem, we then define a state variable ``y``, an adjoint variable
``p`` and a control variable ``u``, and write the PDE as a weak form ::

    y = Function(V)
    p = Function(V)
    u = Function(V)
    e = inner(grad(y), grad(p)) - u*p*dx
    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])

Finally, we define the cost functional and the optimization problem ::

    y_d = Expression('sin(2*pi * x[0] * sin(2*pi*x[1]))', degree=1)
    alpha = 1e-6
    J = 1/2*(y - y_d) * (y - y_d) * dx + alpha/2*u*u*dx
    opt_problem = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    opt_problem.solve()

The only major difference between CASHOCS and FEniCS code is that one has to
use :py:class:`fenics.Function` objects for states and adjoints, and that :py:class:`fenics.TrialFunction`
and :py:class:`fenics.TestFunction` are not needed to define the state equation.
Other than that, the syntax would also be valid with FEniCS, at least for this
problem.

For a detailed discussion of the features of CASHOCS and its usage we refer to the
:ref:`CASHOCS tutorial <tutorial_index>`.
