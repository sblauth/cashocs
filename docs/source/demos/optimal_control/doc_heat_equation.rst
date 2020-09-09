.. _demo_heat_equation:

Distributed Control for Time Dependent Problems
===============================================

Problem Formulation
-------------------

In this demo  we take a look at how time dependent problems can be treated with CASHOCS.
To do so, we investigate a problem considered in
`Blauth, Optimal Control and Asymptotic Analysis of the Cattaneo Model
<https://nbn-resolving.org/urn:nbn:de:hbz:386-kluedo-53727>`_.
It reads

.. math::

    &\min\; J(y,u) = \frac{1}{2} \int_0^T \int_\Omega \left( y - y_d \right)^2 \text{d}x \text{d}t + \frac{\alpha}{2} \int_0^T \int_\Omega u^2 \text{d}x \text{d}t \\
    &\text{ subject to }\quad \left\lbrace \quad
    \begin{alignedat}{2}
    \partial_t y - \Delta y &= u \quad &&\text{ in } (0,T) \times \Omega,\\
    y &= 0 \quad &&\text{ on } (0,T) \times \Gamma, \\
    y(0, \cdot) &= y_0 \quad &&\text{ in } \Omega.
    \end{alignedat} \right.


Since FEniCS does not have any direct built-in support for time dependent problems,
CASHOCS also does not have one. Hence, one first has to perform a semi-discretization
of the PDE system in the temporal component (e.g. via finite differences), and then
solve the resulting sequence of PDEs.

In particular, for the use with CASHOCS, we have to create not a single weak form and
:py:class:`fenics.Function` Function, that can be re-used, like one would in classical FEniCS programs, but
we have to create the corresponding objects a-priori for each time step.

For the domain of this problem, we once again consider the space time cylinder given by :math:`(0,T) \times \Omega = (0,1) \times (0,1)^2`.
And for the initial condition we use :math:`y_0 = 0`.

Temporal discretization
***********************

For the temporal discretization, we use the implicit Euler scheme as this is unconditionally stable for the parabolic heat equation. This means, we discretize the
interval :math:`[0,T]` by a grid with nodes :math:`t_k, k=1,\dots, n,\; \text{ with }\; t_0 := 0\; \text{ and }\; t_n := T`. Then, we approximate the time derivative
:math:`\partial_t y(t_k)` at some time :math:`t_k` by the backward difference

.. math:: \partial_t y(t_k) \approx \frac{y(t_k) - y(t_{k-1})}{\Delta t},

where :math:`\Delta t = t_k - t_{k-1}`, and thus get the sequence of PDEs

.. math::

    \frac{y_k - y_{k-1}}{\Delta t} - \Delta y_k = u_k \quad \text{ in }\; \Omega\; \text{for}\; k=1,\dots,n,\\
    y_k = 0 \quad \text{on}\; \Gamma\; \text{for}\; k=1,\dots,n,


Note, that :math:`y_k \approx y(t_k), \text {and }\; u_k \approx u(t_k)` are approximations of the
continuous functions. The initial condition is included by definition of :math:`y_0`.

Moreover, for the cost functionals, we can discretize the temporal integrals using
rectangle rules. This means we approximate the cost functional via

.. math:: J(y, u) \approx \frac{1}{2} \sum_{k=1}^n \Delta t \left( \int_\Omega \left( y_k - (y_d)_k \right)^2 \text{d}x  + \alpha \int_\Omega u_k^2 \text{d}x \right).


Here, :math:`(y_d)_k` is an approximation of the desired state at time :math:`t_k`.

Let us now investigate how to solve this problem with CASHOCS.


Implementation
--------------
The complete python code can be found in the file :download:`demo_heat_equation.py </../../demos/documented/optimal_control/heat_equation/demo_heat_equation.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/optimal_control/heat_equation/config.ini>`.


Initialization
**************

This section is the same as for all previous problems and is done via ::

    from fenics import *
    import cashocs
    import numpy as np


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    V = FunctionSpace(mesh, 'CG', 1)

Next up, we specify the temporal discretization via ::

    dt = 1 / 10
    t_start = dt
    t_end = 1.0
    t_array = np.linspace(t_start, t_end, int(1/dt))

Here, ``t_array`` is a numpy array containing all time steps. Note, that we do **not**
include t=0 in the array. This is due to the fact, that the initial condition
is prescribed and fixed. Due to the fact that we discretize the equation temporally,
we do not only get a single :py:class:`fenics.Function` describing our state and control, but
one :py:class:`fenics.Function` for each time step. Hence, we initialize these
(together with the adjoint states) directly in lists ::

    states = [Function(V) for i in range(len(t_array))]
    controls = [Function(V) for i in range(len(t_array))]
    adjoints = [Function(V) for i in range(len(t_array))]

Note, that ``states[k]`` corresponds to :math:`y_{k+1}` (due to the differences in indexing between computer scientists and
mathematicians), and analogously for ``controls[k]``. Note, that in the following there
will  be made no such distinctions anymore, it should be obvious from the context
what and where to apply the index shift between the semi-continuous and the discretized
versions of the functions.

As the boundary conditions are not time dependent, we can initialize them now, and
repeat them in a list, since they are the same for every state ::

    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
    bcs_list = [bcs for i in range(len(t_array))]

To define the sequence of PDEs, we will use a loop over all time steps. But before we
can do that, we first initialize empty lists for the state equations, the
approximations of the desired state, and the summands of the cost functional ::

    y_d = []
    e = []
    J_list = []

Definition of the optimization problem
**************************************

For the desired state, we define it with the help of a :py:class:`fenics.Expression`, that is
dependent on an additional parameter which models the time ::

    alpha = 1e-5
    y_d_expr = Expression('exp(-20*(pow(x[0] - 0.5 - 0.25*cos(2*pi*t), 2) + pow(x[1] - 0.5 - 0.25*sin(2*pi*t), 2)))', degree=1, t=0.0)

Next, we have the following for loop, which we describe in detail after stating it here ::

    for k in range(len(t_array)):
    	t = t_array[k]
    	y_d_expr.t = t

    	y = states[k]
    	if k == 0:
    		y_prev = Function(V)
    	else:
    		y_prev = states[k - 1]
    	p = adjoints[k]
    	u = controls[k]

    	state_eq = Constant(1/dt)*(y - y_prev)*p*dx + inner(grad(y), grad(p))*dx - u*p*dx

    	e.append(state_eq)
    	y_d.append(interpolate(y_d_expr, V))

    	J_list.append(Constant(0.5*dt) * (y - y_d[k]) * (y - y_d[k]) * dx + Constant(0.5 * dt * alpha) * u * u * dx)

.. note::

    At the beginning, the 'current' time t is determined from ``t_array``, and the
    expression for the desired state is updated to reflect the current time.
    The line ::

        y = states[k]

    sets the object ``y`` to :math:`y_k`. For the backward difference in the implicit Euler method, we also need
    :math:`y_{k-1}` which we define by the if condition ::

        if k == 0:
            y_prev = Function(V)
        else:
            y_prev = states[k - 1]

    which ensures that :math:`y_0 = 0`. Hence, ``y_prev`` indeed corresponds to :math:`y_{k-1}`.
    Moreover, we get the current control and adjoint state via ::

        p = adjoints[k]
        u = controls[k]

    This allow us to define the state equation at time t as ::

        state_eq = Constant(1/dt)*(y - y_prev)*p*dx + inner(grad(y), grad(p))*dx - u*p*dx

    This is then appended to the list of state constraints ::

        e.append(state_eq)

    Further, we also put the current desired state into the respective list, i.e., ::

        y_d.append(interpolate(y_d_expr, V))

    Finally, we can define the k-th summand of the cost functional via ::

        J_list.append(Constant(0.5*dt) * (y - y_d[k]) * (y - y_d[k]) * dx + Constant(0.5 * dt * alpha) * u * u * dx)

    and directly append this to the cost functional list.

To sum up over all elements of
this list, CASHOCS includes the function :py:func:`cashocs.utils.summation`, which we call ::

    J = cashocs.utils.summation(J_list)

Finally, we can define an optimal control as always, and solve it as in the previous demos (see, e.g., :ref:`demo_poisson`) ::

    ocp = cashocs.OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)
    ocp.solve()

For a postprocessing, which visualizes the resulting optimal control and optimal state,
the following lines are added at the end ::

    u_file = File('./visualization/u.pvd')
    y_file = File('./visualization/y.pvd')
    temp_u = Function(V)
    temp_y = Function(V)

    for k in range(len(t_array)):
    	t = t_array[k]

    	temp_u.vector()[:] = controls[k].vector()[:]
    	u_file << temp_u, t

    	temp_y.vector()[:] = states[k].vector()[:]
    	y_file << temp_y, t

which saves the result in the folder visualization as paraview .pvd files.
