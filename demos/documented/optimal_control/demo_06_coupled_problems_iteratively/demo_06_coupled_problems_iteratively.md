Demo 06 : Coupled Problems Part II - Picard Approach
====================================================

In this demo we show how cashocs can be used with a coupled PDE constraint.
For this demo, we consider a iterative approach, whereas we investigated
a monolithic approach in the previous demo.

As model example, we consider the
following problem

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmin%5C%3B+J%28%28y%2Cz%29%2C%28u%2Cv%29%29+%3D+%5Cfrac%7B1%7D%7B2%7D+%5Cint_%5COmega+%5Cleft%28+y+-+y_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B1%7D%7B2%7D+%5Cint_%5COmega+%5Cleft%28+z+-+z_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B%5Calpha%7D%7B2%7D+%5Cint_%5COmega+u%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B%5Cbeta%7D%7B2%7D+%5Cint_%5COmega+v%5E2+%5Ctext%7Bd%7Dx"
alt="\min\; J((y,z),(u,v)) = \frac{1}{2} \int_\Omega \left( y - y_d \right)^2 \text{d}x + \frac{1}{2} \int_\Omega \left( z - z_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_\Omega u^2 \text{d}x + \frac{\beta}{2} \int_\Omega v^2 \text{d}x">

subject to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A-%5CDelta+y+%2B+z+%26%3D+u+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%2C+%5C%5C%0A-%5CDelta+z+%2B+y+%26%3D+v+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%2C+%5C%5C%0Ay+%26%3D+0+%5Cquad+%5Ctext%7B+on+%7D%5C%3B+%5CGamma%2C+%5C%5C%0Az+%26%3D+0+%5Cquad+%5Ctext%7B+on+%7D%5C%3B+%5CGamma%2C+%5C%5C%0A%5Cend%7Balign%2A%7D"
alt="\begin{align*}
-\Delta y + z &= u \quad \text{ in }\; \Omega, \\
-\Delta z + y &= v \quad \text{ in }\; \Omega, \\
y &= 0 \quad \text{ on }\; \Gamma, \\
z &= 0 \quad \text{ on }\; \Gamma, \\
\end{align*}">

Again, the system is two-way coupled. To solve it, we now employ a Picard iteration. Therefore,
the two PDEs are solved subsequently, where the variables are frozen in between: At the beginning
the first PDE is solved, with the second state variable being fixed. Then, the second PDE is solved
with the value of the first variable fixed (to the one obtained by the prior solve). This is then repeated
until convergence is reached (but of course this does not have to occur).

Initialization
--------------

The setup is as usual

    from fenics import *
    import cashocs


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
    V = FunctionSpace(mesh, 'CG', 1)

However, compared to the previous examples, there is a major change in the config file. As we want to use
the Picard iteration as solver for the state PDEs, we now specify

    picard_iteration = true

in the config file. Note, that only the flag is_linear = false works for the Picard iteration, regardless
of the actual (non)-linearity of the underlying problem.    

Definition of the state equations
---------------------------------

As we solve both PDEs decoupled (or seperately), we now only need a single FunctionSpace object. The
corresponding state and adjoint variables are defined via

    y = Function(V)
    z = Function(V)
    p = Function(V)
    q = Function(V)
    states = [y, z]
    adjoints = [p, q]

which basically reverses the idea of the monolithic approach. Here, we first define the "components" as
single, decoupled functions, and only identify them as the state variables later by putting them
into a joint list. The same is true for the adjoint variables.

The control variables are defined as previously

    u = Function(V)
    v = Function(V)
    controls = [u, v]

Similarly to before, we define the state equations, but instead of adding them, we also put them
into a joint list, since we solve them in a decoupled fashion

    e1 = inner(grad(y), grad(p))*dx + z*p*dx - u*p*dx
    e2 = inner(grad(z), grad(q))*dx + y*q*dx - v*q*dx
    e = [e1, e2]

The boundary conditions are treated analogously

    bcs1 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
    bcs2 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
    bcs = [bcs1, bcs2]

Definition of the optimization problem
--------------------------------------

The same is true for the cost functional

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
    alpha = 1e-6
    beta = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
        + Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

Finally, we set up the optimization problem and solve it

    optimization_problem = cashocs.OptimalControlProblem(e, bcs, J, states, controls, adjoints, config)
    optimization_problem.solve()

> Comparing the output (especially in the early iterations) between the monlithic and Picard apporach
> we observe that both methods yield essentially the same results (up to machine precision). This validates
> the Picard approach.
>
> However, one should note that for this example, the Picard approach takes significantly longer to
> compute the optimizer. This is due to the fact that the individual PDEs have to be solved several
> times, whereas in the monolithic approach the state system is (slightly) larger, but has to be solved
> less often. However, the monolithic approach needs significantly more memory, so that the Picard
> iteration becomes feasible for very large problems. Further, the convergence properties of the
> Picard iteration are better, so that it can converge even when the monolithic approach fails.
