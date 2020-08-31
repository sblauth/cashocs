## Demo 09: Nonlinear PDE constraints

In this demo, we take a look at the case of nonlinear PDE constraints for optimization
problems. As a model problem, we consider

$$\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{d}x \\
\text{ subject to } \quad \left\lbrace \quad
\begin{alignedat}{2}
-\Delta y + c y^3 &= u \quad &&\text{ in } \Omega,\\
y &= 0 \quad &&\text{ on } \Gamma.
\end{alignedat} \right.
$$

As this problem has a nonlinear PDE as state constraint, we have to modify the config
file slightly. In particular, in the section `StateEquation` we have to write

    is_linear = false

Note, that `is_linear = false` always works, as linear equations are just a special case
of nonlinear ones, and the Newton method converges in a single iteration for these.
However, in the opposite case, FEniCS will raise some error, and a real nonlinear
equation cannot be solved using `is_linear = true`.

**Initialization**

Thanks to the high level interface for implementing weak formulations, this problem
is tackled almost as easily as our first one. In particular, the entire initialization,
up to the definition of the weak form of the PDE constraint, is identical, and we have

    from fenics import *
    import cashocs


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

For the definition of the state constraints, we use essentially the same syntax as
we would use for the problem in FEniCS, i.e., we write

    c = Constant(1e2)
    e = inner(grad(y), grad(p))*dx + c*pow(y, 3)*p*dx - u*p*dx

> In particular, the only difference between the cashocs implementation of this weak form
> and the FEniCS one is that, as before, we use `Function` objects for both the state and
> adjoint variables, whereas we would use `Function` objects for the state, and
> `TestFunction` for the "adjoint" variable, which would actually play the role of the
> test function. Other than that, the syntax is, again, identical.

Finally, the boundary conditions are defined as before

    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

**Solution of the optimization problem**

To define and solve the optimization problem, we now proceed exactly as before, and use

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    alpha = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    ocp.solve()
