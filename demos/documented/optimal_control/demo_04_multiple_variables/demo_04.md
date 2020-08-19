Demo 04 : Multiple Variables
============================

In this demo we show how cestrel can be used to treat multiple
state equations as constraint. Additionally, this also highlights 
how multiple controls can be treated. As model example, we consider the 
following problem

min J(y, u) = 1/2 || y<sub>1</sub> - (y<sub>1</sub>)<sub>d</sub> ||<sub>&Omega;</sub><sup>2</sup> 
              + 1/2 || y<sub>2</sub> - (y<sub>2</sub>)<sub>d</sub> ||<sub>&Omega;</sub><sup>2</sup> 
              + &alpha;/2  || u ||<sub>&Omega;</sub><sup>2</sup>
              + &beta;/2  || v ||<sub>&Omega;</sub><sup>2</sup>

subject to &nbsp;&nbsp;&nbsp;  - &nabla; &middot; ( &nabla; y<sub>1</sub>  ) = u &nbsp;&nbsp;&nbsp;&nbsp; in &Omega;

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nabla; &middot; ( &nabla; y<sub>2</sub>  ) - y<sub>1</sub> = v &nbsp;&nbsp;&nbsp;&nbsp; in &Omega;

 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y<sub>1</sub> = 0 &nbsp;&nbsp;&nbsp;&nbsp; on &Gamma;
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y<sub>2</sub> = 0 &nbsp;&nbsp;&nbsp;&nbsp; on &Gamma;

For the sake of simplicity, we restrict this investigation to
homogeneous boundary conditions as well as to a very simple one way
coupling. More complex problems (using e.g. Neumann control or more
difficult couplings) are straightforward to implement.

In contrast to the previous examples, in the case where we have multiple state equations, which are
either decoupled or only one-way coupled, the corresponding state equations are solved one after the other
so that every input related to the state and adjoint variables has to be put into a ordered list, so
that they can be treated subsequently.

Initialization
--------------

The initial setup is identical to the previous cases, where we again use

    from fenics import *
    import cestrel
    
    
    
    set_log_level(LogLevel.CRITICAL)
    config = cestrel.create_config('./config.ini')
    
    mesh, subdomains, boundaries, dx, ds, dS = cestrel.regular_mesh(50)
    V = FunctionSpace(mesh, 'CG', 1)

which defines the geometry and the function space.

Defintion of the variables
--------------------------

Next, we have to define the state, adjoint, and control variables, which 
we do with

    y1 = Function(V)
    y2 = Function(V)
    p1 = Function(V)
    p2 = Function(V)
    u = Function(V)
    v = Function(V)

Here p1 is the adjoint state corresponding to y1, and p2 is the adjoint 
state corresponding to y2. For the treatment with cestrel these have to 
be put in (ordered) lists, so that the states and adjoints obey the
same order. This means, we define

    y = [y1, y2]
    p = [p1, p2]
    controls = [u, v]

Note, that the control variables are completely independent of the state
and adjoint ones, so that the relative ordering between these objects does 
not matter. 

Defintion of the state equations / state system
-----------------------------------------------

Now, we can define the PDE constraints corresponding to y1 and y2, which
read in fenics syntax

    e1 = inner(grad(y1), grad(p1))*dx - u*p1*dx
    e2 = inner(grad(y2), grad(p2))*dx - (y1 + v)*p2*dx

Again, the state equations have to be gathered into a list, where the order
has to be in analogy to the list y, i.e.,

    pdes = [e1, e2]

Finally, the boundary conditions for both states are homogeneous 
Dirichlet conditions, which we generate via

    bcs1 = cestrel.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
    bcs2 = cestrel.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
    
    bcs_list = [bcs1, bcs2]
    
and who are also put into a joint list.

Defintion of the cost functional and optimization problem
---------------------------------------------------------

For the optimization problem we now define the cost functional via

    y1_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    y2_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)
    alpha = 1e-6
    beta = 1e-4
    J = Constant(0.5)*(y1 - y1_d)*(y1 - y1_d)*dx + Constant(0.5)*(y2 - y2_d)*(y2 - y2_d)*dx \
        + Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx
        
This setup is sufficient to now define the optimal control problem and solve
it, via

    optimization_problem = cestrel.OptimalControlProblem(pdes, bcs_list, J, y, controls, p, config)
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
>     scalar_product_u = ...
>     scalar_product_v = ...
>     scalar_products = [scalar_product_u, scalar_produt_v]
>

In summary, to treat multiple (control or state) variables, the 
corresponding objects simply have to placed into ordered lists which
are then given to the OptimalControlProblem instead of the "single" objects.
Note, that each individual object of these lists is allowed to be from a
different function space.
