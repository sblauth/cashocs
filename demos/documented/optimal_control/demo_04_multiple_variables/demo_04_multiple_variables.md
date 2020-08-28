## Demo 04 : Multiple Variables


In this demo we show how cashocs can be used to treat multiple
state equations as constraint. Additionally, this also highlights
how multiple controls can be treated. As model example, we consider the
following problem

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmin%5C%3B+J%28%28y%2Cz%29%2C%28u%2Cv%29%29+%3D+%5Cfrac%7B1%7D%7B2%7D+%5Cint_%5COmega+%5Cleft%28+y+-+y_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B1%7D%7B2%7D+%5Cint_%5COmega+%5Cleft%28+z+-+z_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B%5Calpha%7D%7B2%7D+%5Cint_%5COmega+u%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B%5Cbeta%7D%7B2%7D+%5Cint_%5COmega+v%5E2+%5Ctext%7Bd%7Dx"
alt="\min\; J((y,z),(u,v)) = \frac{1}{2} \int_\Omega \left( y - y_d \right)^2 \text{d}x + \frac{1}{2} \int_\Omega \left( z - z_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_\Omega u^2 \text{d}x + \frac{\beta}{2} \int_\Omega v^2 \text{d}x">

subject to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A-%5CDelta+y+%26%3D+u+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%2C+%5C%5C%0A-%5CDelta+z+-+y+%26%3D+v+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%2C+%5C%5C%0Ay+%26%3D+0+%5Cquad+%5Ctext%7B+on+%7D%5C%3B+%5CGamma%2C+%5C%5C%0Az+%26%3D+0+%5Cquad+%5Ctext%7B+on+%7D%5C%3B+%5CGamma%2C+%5C%5C%0A%5Cend%7Balign%2A%7D"
alt="\begin{align*}
-\Delta y &= u \quad \text{ in }\; \Omega, \\
-\Delta z - y &= v \quad \text{ in }\; \Omega, \\
y &= 0 \quad \text{ on }\; \Gamma, \\
z &= 0 \quad \text{ on }\; \Gamma, \\
\end{align*}">

For the sake of simplicity, we restrict this investigation to
homogeneous boundary conditions as well as to a very simple one way
coupling. More complex problems (using e.g. Neumann control or more
difficult couplings) are straightforward to implement.

In contrast to the previous examples, in the case where we have multiple state equations, which are
either decoupled or only one-way coupled, the corresponding state equations are solved one after the other
so that every input related to the state and adjoint variables has to be put into a ordered list, so
that they can be treated subsequently.

**Initialization**


The initial setup is identical to the previous cases, where we again use

    from fenics import *
    import cashocs


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
    V = FunctionSpace(mesh, 'CG', 1)

which defines the geometry and the function space.

**Defintion of the variables**

Next, we have to define the state, adjoint, and control variables, which
we do with

    y = Function(V)
    z = Function(V)
    p = Function(V)
    q = Function(V)
    u = Function(V)
    v = Function(V)

Here `p` is the adjoint state corresponding to `y`, and `q` is the adjoint
state belonging to `z`. For the treatment with cashocs these have to
be put in (ordered) lists, so that the states and adjoints obey the
same order. This means, we define

    states = [y, z]
    adjoints = [p, q]
    controls = [u, v]

Note, that the control variables are completely independent of the state
and adjoint ones, so that the relative ordering between these objects does
not matter.

**Defintion of the state equations / state system**


Now, we can define the PDE constraints corresponding to `y` and `z`, which
read in fenics syntax

    e_y = inner(grad(y), grad(p)) * dx - u * p * dx
    e_z = inner(grad(z), grad(q)) * dx - (y + v) * q * dx

Again, the state equations have to be gathered into a list, where the order
has to be in analogy to the list y, i.e.,

    e = [e_y, e_z]

Finally, the boundary conditions for both states are homogeneous
Dirichlet conditions, which we generate via

    bcs1 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
    bcs2 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

    bcs_list = [bcs1, bcs2]

and who are also put into a joint list.

**Defintion of the cost functional and optimization problem**


For the optimization problem we now define the cost functional via

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
    alpha = 1e-6
    beta = 1e-4
    J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5) * (z - z_d) * (z - z_d) * dx \
    	+ Constant(0.5*alpha) * u * u * dx + Constant(0.5*beta) * v * v * dx

This setup is sufficient to now define the optimal control problem and solve
it, via

    optimization_problem = cashocs.OptimalControlProblem(e, bcs_list, J, states, controls, adjoints, config)
    optimization_problem.solve()

> Note, that for the case that we consider control constraints (see demo_02)
> or different Hilbert spaces, e.g., for boundary control (see demo_03),
> the corresponding control constraints have also to be put into a list, i.e.,
>
>     cc_u = [u_a, u_b]
>     cc_v = [v_a, v_b]
>     cc = [cc_u, cc_v]
>
> and the corresponding scalar products are treated analogously, i.e.,
>
>     scalar_product_u = TrialFunction(V)*TestFunction(V)*dx
>     scalar_product_v = TrialFunction(V)*TestFunction(V)*dx
>     scalar_products = [scalar_product_u, scalar_produt_v]
>

In summary, to treat multiple (control or state) variables, the
corresponding objects simply have to placed into ordered lists which
are then given to the OptimalControlProblem instead of the "single" objects.
Note, that each individual object of these lists is allowed to be from a
different function space.
