## Demo 03 : Neumann boundary control


In this demo we investigate an optimal control problem with
a Neumann type boundary control. This problem reads

$$\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Gamma} u^2 \text{d}s \\
\text{ subject to } \quad \left\lbrace \quad
\begin{alignedat}{2}
-\Delta y &= 0 \quad &&\text{ in } \Omega,\\
n\cdot \nabla y &= u \quad &&\text{ on } \Gamma.
\end{alignedat} \right.
$$

(see, e.g., [Tröltzsch, Optimal Control of Partial Differential Equations](https://doi.org/10.1090/gsm/112),
or [Hinze et al., Optimization with PDE constraints](https://doi.org/10.1007/978-1-4020-8839-1)).
Note, that we cannot use a simple Poisson equation as constraint
since this would not be compatible with the boundary conditions
(i.e. not well-posed).

**Initialization**

Initially, the code is again identical to the one demo_01 and demo_02,
i.e., we have

    from fenics import *
    import cashocs


    config = cashocs.create_config('./config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

**Definition of the state equation**


Now, the definition of the state problem obviously differs from the
previous two examples, and we now use

    e = inner(grad(y), grad(p))*dx + y*p*dx - u*p*ds

which directly puts the Neumann boundary condition into the weak form.
For this problem, we do not have Dirichlet boundary conditions, so that we
use

    bcs = None

> Alternatively, we could have also used a empty list, i.e.,
>
>     bcs = []
>
> instead

**Definition of the cost functional**


The definition of the cost functional is now nearly identical to before,
only the integration measure for the regularization term changes, so that we have

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    alpha = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*ds

As the default Hilbert space for a control is \( L^2(\Omega) \), we now
also have to change this, to accommodate for the fact that the control
variable u now lies in the space \( L^2(\Gamma) \), i.e., it is
only defined on the boundary. This is done by defining the scalar
product of the corresponding Hilbert space, which we do with

    scalar_product = TrialFunction(V)*TestFunction(V)*ds

The scalar_product always has to be a symmetric, coercive and continuous
bilinear form, so that it induces an actual scalar product on the
corresponding space.

**Setup of the optimization problem and its solution**


With this, we can now define the optimal control problem with the
additional keyword argument riesz_scalar_products as follows

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config, riesz_scalar_products=scalar_product)
    ocp.solve()

which also directly solves the problem.

Hence, in order to treat boundary control problems, the corresponding
weak forms have to be modified accordingly, and one **has to** adapt the
scalar products used to determine the gradients.
