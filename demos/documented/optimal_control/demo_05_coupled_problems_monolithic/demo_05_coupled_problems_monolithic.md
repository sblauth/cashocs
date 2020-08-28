## Demo 05 : Coupled Problems Part I - Monolithic Approach


In this demo we show how cashocs can be used with a coupled PDE constraint.
For this demo, we consider a monolithic approach, whereas we investigate
an approach based on a Picard iteration in the following demo.

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

In constrast to the example in demo_04, the system is now two-way coupled. To solve it, we employ a mixed finite element method in this demo.

**Initialization and variable definitions**


The initialization for this example works as before, i.e., we use

    from fenics import *
    import cashocs


    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(50)

For the mixed finite element method we have to define a "mixed" function space, via

    elem_1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    elem_2 = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, MixedElement([elem_1, elem_2]))

The control variables get their own function space

    U = FunctionSpace(mesh, 'CG', 1)

Then, the state and adjoint variables are defined

    state = Function(V)
    adjoint = Function(V)
    y, z = split(state)
    p, q = split(adjoint)

Here, the `split` command allows us to acces the individual components of the elements, which is very
helpful for defining the mixed weak form in the following.

We then define the control variables as

    u = Function(U)
    v = Function(U)
    controls = [u, v]

and group them to the list controls.

> An alternative way of specifying the controls would be to reuse the mixed function space and use
>
>     controls = Function(V)
>     u, v = split(controls)
>
> Allthough this formulation is slightly different (it uses a Function for the controls, and not a list)
> the de-facto behavior of both methods is completely identical, just the interpretation is slightly
> different (since the individual components of the V FunctionSpace are also CG1 functions).

**Definition of the mixed weak form**


Next, we define the mixed weak form, by specifying the components individually and then summing them up

    e1 = inner(grad(y), grad(p))*dx + z*p*dx - u*p*dx
    e2 = inner(grad(z), grad(q))*dx + y*q*dx - v*q*dx
    e = e1 + e2

Note, that we can only have one state equation as we also have only a single state variable `state`,
and the number of state variables and state equations has to coincide.

Moreover, we define the boundary conditions for the components as

    bcs1 = cashocs.create_bcs_list(V.sub(0), Constant(0), boundaries, [1,2,3,4])
    bcs2 = cashocs.create_bcs_list(V.sub(1), Constant(0), boundaries, [1,2,3,4])
    bcs = bcs1 + bcs2

Again, note that we now return a single list of DirichletBC objects, since both lists specify the boundary
conditions for the components of `state`.

**Defintion of the optimization problem**


The cost functional can be specified in analogy to the previous one

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
    alpha = 1e-6
    beta = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
        + Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

Finally, we can set up the optimization problem and solve it

    optimization_problem = cashocs.OptimalControlProblem(e, bcs, J, state, controls, adjoint, config)
    optimization_problem.solve()
