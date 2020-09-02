## Demo 06 : Coupled Problems Part II - Picard Approach


In this demo we show how cashocs can be used with a coupled PDE constraint.
For this demo, we consider a iterative approach, whereas we investigated
a monolithic approach in the previous demo.

As model example, we consider the
following problem

$$\min\; J((y,z),(u,v)) = \frac{1}{2} \int_\Omega \left( y - y_d \right)^2 \text{d}x + \frac{1}{2} \int_\Omega \left( z - z_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_\Omega u^2 \text{d}x + \frac{\beta}{2} \int_\Omega v^2 \text{d}x \\
\text{ subject to }\quad \left\lbrace \quad
\begin{alignedat}{2}
-\Delta y + z &= u \quad &&\text{ in } \Omega, \\
-\Delta z + y &= v \quad &&\text{ in } \Omega,\\
y &= 0 \quad &&\text{ on } \Gamma,\\
z &= 0 \quad &&\text{ on } \Gamma.
\end{alignedat} \right.
$$

Again, the system is two-way coupled. To solve it, we now employ a Picard iteration. Therefore,
the two PDEs are solved subsequently, where the variables are frozen in between: At the beginning
the first PDE is solved, with the second state variable being fixed. Then, the second PDE is solved
with the value of the first variable fixed (to the one obtained by the prior solve). This is then repeated
until convergence is reached (but of course this does not have to occur).

**Initialization**

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

**Definition of the state equations**


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

**Definition of the optimization problem**


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

The result should look like this

![](./img/optimal_control/06_picard_iteration.png)

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
