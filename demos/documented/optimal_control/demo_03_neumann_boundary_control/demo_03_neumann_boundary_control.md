## Demo 03 : Neumann boundary control


In this demo we investigate an optimal control problem with
a Neumann type boundary control. This problem reads

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmin%5C%3B+J%28y%2Cu%29+%3D+%5Cfrac%7B1%7D%7B2%7D+%5Cint_%5COmega+%5Cleft%28+y+-+y_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B%5Calpha%7D%7B2%7D+%5Cint_%5CGamma+u%5E2+%5Ctext%7Bd%7Ds"
alt="\min\; J(y,u) = \frac{1}{2} \int_\Omega \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_\Gamma u^2 \text{d}s">

subject to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A-%5CDelta+y+%2B+y+%26%3D+0+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%2C+%5C%5C%0An+%5Ccdot+%5Cnabla+y+%26%3D+u+%5Cquad+%5Ctext%7B+on+%7D%5C%3B+%5CGamma%0A%5Cend%7Balign%2A%7D"
alt="\begin{align*}
-\Delta y + y &= 0 \quad \text{ in }\; \Omega, \\
n \cdot \nabla y &= u \quad \text{ on }\; \Gamma
\end{align*}">

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

As the default Hilbert space for a control is L<sup>2</sup>(&Omega;), we now
also have to change this, to accommodate for the fact that the control
variable u now lies in the space L<sup>2</sup>(&Gamma;), i.e., it is
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
