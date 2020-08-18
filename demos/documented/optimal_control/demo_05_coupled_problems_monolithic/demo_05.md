Demo 05 : Coupled Problems Part I - Monolithic Approach
=======================================================

In this demo we show how descendal can be used with a coupled PDE constraint. 
For this demo, we consider a monolithic approach, whereas we investigate 
an approach based on a Picard iteration in the following demo.

As model example, we consider the 
following problem

min J(y, u) = 1/2 || y - y<sub>d</sub> ||<sub>&Omega;</sub><sup>2</sup> 
              + 1/2 || z - z<sub>d</sub> ||<sub>&Omega;</sub><sup>2</sup> 
              + &alpha;/2  || u ||<sub>&Omega;</sub><sup>2</sup>
              + &beta;/2  || v ||<sub>&Omega;</sub><sup>2</sup>

subject to &nbsp;&nbsp;&nbsp;  - &nabla; &middot; ( &nabla; y  ) + z = u &nbsp;&nbsp;&nbsp;&nbsp; in &Omega;

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nabla; &middot; ( &nabla; z  ) + y = v &nbsp;&nbsp;&nbsp;&nbsp; in &Omega; 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y = 0 &nbsp;&nbsp;&nbsp;&nbsp; on &Gamma;

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; z = 0 &nbsp;&nbsp;&nbsp;&nbsp; on &Gamma;

In constrast to the example in demo_04, the system is two-way coupled. To solve it, we can employ a 
mixed finite element method. 

Initialization and variable definitions
---------------------------------------

The initialization for this example works as before, i.e., we use

    from fenics import *
    import descendal
    
    
    
    set_log_level(LogLevel.CRITICAL)
    config = descendal.create_config('config.ini')
    
    mesh, subdomains, boundaries, dx, ds, dS = descendal.regular_mesh(50)
   
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
   
Here, the split command allows us to acces the individual components of the elements, which is very
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

Definition of the mixed weak form
---------------------------------

Next, we define the mixed weak form, by specifying the components individually and then summing them up

    e1 = inner(grad(y), grad(p))*dx + z*p*dx - u*p*dx
    e2 = inner(grad(z), grad(q))*dx + y*q*dx - v*q*dx
    e = e1 + e2

Note, that we can only have one state equation as we also have only a single state variable, namely "state",
and the number of state variables and state equations has to coincide.

Moreover, we define the boundary conditions for the components as

    bcs1 = descendal.create_bcs_list(V.sub(0), Constant(0), boundaries, [1,2,3,4])
    bcs2 = descendal.create_bcs_list(V.sub(1), Constant(0), boundaries, [1,2,3,4])
    bcs = bcs1 + bcs2

Again, note that we now return a single list of DirichletBC objects, since both lists specify the boundary
conditions for the components of "state".

Defintion of the optimization problem
-------------------------------------

The cost functional can be specified in analogy to the previous one

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
    alpha = 1e-6
    beta = 1e-6
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
        + Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

Finally, we can set up the optimization problem and solve it

    optimization_problem = descendal.OptimalControlProblem(e, bcs, J, state, controls, adjoint, config)
    optimization_problem.solve()
    
