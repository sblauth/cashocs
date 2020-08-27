Demo 02 : Control Constraints
=============================

In this demo we investigate the basics of cashocs for
optimal control problems. To do so, we investigate the "mother
problem" of PDE constrained optimization, i.e.,

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmin%5C%3B+J%28y%2Cu%29+%3D+%5Cfrac%7B1%7D%7B2%7D+%5Cint_%7B%5COmega%7D+%5Cleft%28+y+-+y_d+%5Cright%29%5E2+%5Ctext%7Bd%7Dx+%2B+%5Cfrac%7B%5Calpha%7D%7B2%7D+%5Cint_%7B%5COmega%7D+u%5E2+%5Ctext%7Bd%7Dx"
alt="\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{d}x">

subject to <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A-%5CDelta+y+%26%3D+u+%5Cquad+%5Ctext%7B+in+%7D%5C%3B+%5COmega%2C%5C%5C%0Ay+%26%3D+0+%5Cquad+%5Ctext%7B+on+%7D%5C%3B+%5CGamma%0A%5Cend%7Balign%2A%7D%0A"
alt="\begin{align*}
-\Delta y &= u \quad \text{ in }\; \Omega,\\
y &= 0 \quad \text{ on }\; \Gamma
\end{align*}
">

and <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+u_a+%5Cleq+u+%5Cleq+u_b"
alt="u_a \leq u \leq u_b">

(see, e.g., [Tröltzsch, Optimal Control of Partial Differential Equations](https://doi.org/10.1090/gsm/112),
or [Hinze et al., Optimization with PDE constraints](https://doi.org/10.1007/978-1-4020-8839-1)).

This example differs from the first one only in the fact that
we now also consider box constraints on the control variables
Here, the functions u<sub>a</sub> and u<sub>b</sub> are
<img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+L%5E%5Cinfty%28%5COmega%29"
alt="L^\infty(\Omega)"> functions. As before, we consider
as domain the unit square, i.e., <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5COmega+%3D+%280%2C+1%29+%5Ctimes+%280%2C1%29"
alt="\Omega = (0, 1) \times (0,1)">.

Initialization
--------------

The beginning of the script is completely identical to the
one of the previous example, so we only restate the corresponding
code in the following. A detailed description can be found
in the documentation of "demo_01.py".

    from fenics import *
    import cashocs


    config = cashocs.create_config('./config.ini')

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

    e = inner(grad(y), grad(p))*dx - u*p*dx

    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    alpha = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

Definition of the control constraints
-------------------------------------

Here, we have nearly everything at hand to define the optimal
control problem, the only missing ingredient are the box constraints,
which we define now. For the purposes of this example, we
consider a linear (in the x-direction) corridor for these
constraints, as it highlights the capabilities of the code.
Hence, we define the lower and upper bounds via

    u_a = interpolate(Expression('50*(x[0]-1)', degree=1), V)
    u_b = interpolate(Expression('50*x[0]', degree=1), V)

which just corresponds to two functions, generated from
Expression objects via interpolation. These are then put
into the list `cc`, which models the control constraints, i.e.,

    cc = [u_a, u_b]

Note, that we discuss alternative methods of defining the box
constraints at the end of this documentation.

Setup of the optimization problem and its solution
--------------------------------------------------

Now, we can set up the optimal control problem as we did before,
using the additional keyword argument control_constraints into which
we put the list `cc`, and then solve it via `ocp.solve()`

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config, control_constraints=cc)
    ocp.solve()

To check that the box constraints are actually satisfied by our
solution, we perform an assertion

    import numpy as np
    assert np.alltrue(u_a.vector()[:] <= u.vector()[:]) and np.alltrue(u.vector()[:] <= u_b.vector()[:])

which shows that they are indeed satisfied.

> As an alternative way of specifying the box constraints, one
> can also use regular float or int objects, in case that they
> are constant. For example, the constraint that we only want to
> consider positive value for u, i.e., 0 &le; u &le; &infin; can
> be realized via
>
>     u_a = 0
>     u_b = float(inf)
>     cc = [u_a, u_b]
>
> and completely analogous with float(-inf) for no constraint
> on the lower bound.
