Demo 08 : Heat Equation (Time Dependent Problems)
=================================================

In this demo  we take a look at how time dependent problems can be treated with cashocs.
To do so, we investigate a problem considered in [Blauth, Optimal Control and Asymptotic Analysis of the Cattaneo Model](https://nbn-resolving.org/urn:nbn:de:hbz:386-kluedo-53727) (my Master's thesis). It reads

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmin%5C%3B+J%28y%2Cu%29+%3D+%5Cfrac%7B1%7D%7B2%7D+%5Cint_0%5ET+%5Cint_%5COmega+%5Cleft%28+y+-+y_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%5Ctext%7Bd%7Dt+%2B+%5Cfrac%7B%5Calpha%7D%7B2%7D+%5Cint_0%5ET+%5Cint_%5COmega+u%5E2+%5Ctext%7Bd%7Dx+%5Ctext%7Bd%7Dt"
alt="\min\; J(y,u) = \frac{1}{2} \int_0^T \int_\Omega \left( y - y_d \right)^2 \text{d}x \text{d}t + \frac{\alpha}{2} \int_0^T \int_\Omega u^2 \text{d}x \text{d}t">

subject to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cpartial_t+y+-+%5CDelta+y+%26%3D+u+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%280%2C+T%29+%5Ctimes+%5COmega%2C%5C%5C%0Ay+%26%3D+0+%5Cquad+%5Ctext%7B+on+%7D+%280%2C+T%29+%5Ctimes+%5CGamma%2C%5C%5C%0Ay%280%2C+%5Ccdot%29+%26%3D+y_0+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%0A%5Cend%7Balign%2A%7D"
alt="\begin{align*}
\partial_t y - \Delta y &= u \quad \text{ in }\; (0, T) \times \Omega,\\
y &= 0 \quad \text{ on } (0, T) \times \Gamma,\\
y(0, \cdot) &= y_0 \quad \text{ in }\; \Omega
\end{align*}">

Since fenics does not have any direct built-in support for time dependent problems,
cashocs also does not have one. Hence, one first has to perform a semi-discretization
of the PDE system in the temporal component (e.g. via finite differences), and then
solve the resulting sequence of PDEs.

In particular, for the use with cashocs, we have to create not a single weak form and
fenics Function, that can be re-used, like one would in classical fenics programs, but
we have to create the corresponding objects a-priori for each time step.

For the domain of this problem, we once again consider the space time cylinder given by
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%280%2C+T%29+%5Ctimes+%5COmega"
alt="(0, T) \times \Omega"> <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5COmega+%3D+%280%2C+1%29%5E2"
alt="\Omega = (0, 1)^2">. And for the initial condition we use <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_0+%3D+0"
alt="y_0 = 0">.


Temporal Discretization
-----------------------

For the temporal discretization, we use the implicit Euler scheme as this is unconditionally stable for the parabolic heat equation. This means, we discretize the
interval <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5B0%2CT%5D"
alt="[0,T]"> by a grid with nodes <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+t_k%2C+k%3D1%2C%5Cdots%2C+n%2C%5C%3B+%5Ctext%7B+with+%7D%5C%3B+t_0+%3A%3D+0%5C%3B+%5Ctext%7B+and+%7D%5C%3B+t_n+%3A%3D+T"
alt="t_k, k=1,\dots, n,\; \text{ with }\; t_0 := 0\; \text{ and }\; t_n := T">. Then, we approximate the time derivative <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cpartial_t+y%28t_k%29"
alt="\partial_t y(t_k)"> at some time <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+t_k"
alt="t_k"> by the backward difference

<img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cpartial_t+y%28t_k%29+%5Capprox+%5Cfrac%7By%28t_k%29+-+y%28t_%7Bk-1%7D%29%7D%7B%5CDelta+t%7D"
alt="\partial_t y(t_k) \approx \frac{y(t_k) - y(t_{k-1})}{\Delta t}">,

where <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5CDelta+t+%3D+t_k+-+t_%7Bk-1%7D"
alt="\Delta t = t_k - t_{k-1}">, and thus get the sequence of PDEs

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cfrac%7By_k+-+y_%7Bk-1%7D%7D%7B%5CDelta+t%7D+-+%5CDelta+y_k+%26%3D+u_k+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%5C%3B+%5Ctext%7Bfor%7D%5C%3B+k%3D1%2C%5Cdots%2Cn%2C%5C%5C%0Ay_k+%26%3D+0+%5Cquad+%5Ctext%7Bon%7D%5C%3B+%5CGamma%5C%3B+%5Ctext%7Bfor%7D%5C%3B+k%3D1%2C%5Cdots%2Cn%2C%5C%5C%0A%5Cend%7Balign%2A%7D%0A"
alt="\begin{align*}
\frac{y_k - y_{k-1}}{\Delta t} - \Delta y_k &= u_k \quad \text{ in }\; \Omega\; \text{for}\; k=1,\dots,n,\\
y_k &= 0 \quad \text{on}\; \Gamma\; \text{for}\; k=1,\dots,n,\\
\end{align*}
">.

Note, that <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+y_k+%5Capprox+y%28t_k%29%2C+%5Ctext+%7Band+%7D%5C%3B+u_k+%5Capprox+u%28t_k%29"
alt="y_k \approx y(t_k), \text {and }\; u_k \approx u(t_k)"> are approximations of the
continuous functions. The initial condition is included by definition of <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_0"
alt="y_0">.

Moreover, for the cost functionals, we can discretize the temporal integrals using
rectangle rules. This means we approximate the cost functional via

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J%28y%2C+u%29+%5Capprox+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bk%3D1%7D%5En+%5CDelta+t+%5Cleft%28+%5Cint_%5COmega+%5Cleft%28+y_k+-+%28y_d%29_k+%5Cright%29%5E2+%5Ctext%7Bd%7Dx++%2B+%5Calpha+%5Cint_%5COmega+u_k%5E2+%5Ctext%7Bd%7Dx+%5Cright%29"
alt="J(y, u) \approx \frac{1}{2} \sum_{k=1}^n \Delta t \left( \int_\Omega \left( y_k - (y_d)_k \right)^2 \text{d}x  + \alpha \int_\Omega u_k^2 \text{d}x \right)">.

Here, <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%28y_d%29_k+%5Capprox+y_d%28t_k%29"
alt="(y_d)_k \approx y_d(t_k)"> is an approximation of the desired state at time t_k.

Let us now investigate how to solve this problem with cashocs.

Initialization
--------------

This section is the same as for all previous problems and is done via

    from fenics import *
    import cashocs
    import numpy as np


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    V = FunctionSpace(mesh, 'CG', 1)

Next up, we specify the temporal discretization via

    dt = 1 / 10
    t_start = dt
    t_end = 1.0
    t_array = np.linspace(t_start, t_end, int(1/dt))

Here, `t_array` is a numpy array containing all time steps. Note, that we do **not**
include t=0 in the array. This is due to the fact, that the initial condition
is prescribed and fixed. Due to the fact that we discretize the equation temporally,
we do not only get a single fenics Function describing our state and control, but
one Function for each time step. Hence, we initialize these (together with the adjoint states) directly in lists

    states = [Function(V) for i in range(len(t_array))]
    controls = [Function(V) for i in range(len(t_array))]
    adjoints = [Function(V) for i in range(len(t_array))]

Note, that `states[k]` corresponds to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_%7Bk%2B1%7D"
alt="y_{k+1}"> (due to the differences in indexing between computer scientists and
mathematicians), and analogously for `controls[k]`. Note, that in the following there
will  be made no such distinctions anymore, it should be obvious from the context
what and where to apply the index shift between the semi-continuous and the discretized
versions of the functions.

As the boundary conditions are not time dependent, we can initialize them now, and
repeat them in a list, since they are the same for every state

    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
    bcs_list = [bcs for i in range(len(t_array))]

To define the sequence of PDEs, we will use a loop over all time steps. But before we
can do that, we first initialize empty lists for the state equations, the
approximations of the desired state, and the summands of the cost functional.

    y_d = []
    e = []
    J_list = []

Definition of the optimization problem
--------------------------------------

For the desired state, we define it with the help of a fenics expression, that is
dependent on an additional parameter which models the time.

    alpha = 1e-5
    y_d_expr = Expression('exp(-20*(pow(x[0] - 0.5 - 0.25*cos(2*pi*t), 2) + pow(x[1] - 0.5 - 0.25*sin(2*pi*t), 2)))', degree=1, t=0.0)

Next, we have the following for loop, which we describe in detail in the following

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

> At the beginning, the 'current' time t is determined from `t_array`, and the
> expression for the desired state is updated to reflect the current time.
> The line
>
>     y = states[k]
>
> sets the object `y` to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_k"
alt="y_k">. For the backward difference in the implicit Euler method, we also need
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_%7Bk-1%7D"
alt="y_{k-1}">, which we define by the if condition
>
>     if k == 0:
>         y_prev = Function(V)
>     else:
>         y_prev = states[k - 1]
>
> which ensures that <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_0+%3D+0"
alt="y_0 = 0">. Hence, `y_prev` indeed corresponds to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+y_%7Bk-1%7D"
alt="y_{k-1}">. Moreover, we get the current control and adjoint state via
>
>     p = adjoints[k]
>	u = controls[k]
>
> This allow us to define the state equation at time t as
>
>     state_eq = Constant(1/dt)*(y - y_prev)*p*dx + inner(grad(y), grad(p))*dx - u*p*dx
>
> This is then appended to the list of state constraints
>
>     e.append(state_eq)
>
> Further, we also put the current desired state into the respective list, i.e.,
>
> 	y_d.append(interpolate(y_d_expr, V))
>
> Finally, we can define the k-th summand of the cost functional via
>
>     J_list.append(Constant(0.5*dt) * (y - y_d[k]) * (y - y_d[k]) * dx + Constant(0.5 * dt * alpha) * u * u * dx)
>
> and directly append this to the cost functional list.

To sum up over all elements of
this list, cashocs includes a summation command in the utils module, which we call now

    J = cashocs.utils.summation(J_list)

Finally, we can define an optimal control as always, and solve it in the same fashion

    ocp = cashocs.OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)
    ocp.solve()

For a postprocessing, which visualizes the resulting optimal control and optimal state
the following lines are added at the end

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
