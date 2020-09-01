## Demo 01 : Shape Poisson

In this demo, we investigate the basics of cashocs for shape optimization problems.
As a model problem, we investigate the following one from [Etling et al., First and Second Order Shape Optimization Based on Restricted Mesh Deformations](https://doi.org/10.1137/19M1241465)

$$\min_\Omega J(u, \Omega) = \int_\Omega u \text{d}x \\
\text{subject to} \quad \left\lbrace \quad
\begin{alignedat}{2}
-\Delta u &= f \quad &&\text{ in } \Omega,\\
u &= 0 \quad &&\text{ on } \Gamma.
\end{alignedat} \right.
$$

For the initial domain, we use the unit disc \( \Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \} \), and the right-hand side \(f \) is given by

$$ f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.
$$


**Initialization**

Similarly to the optimal control case, we also require config files for shape optimization problems in cashocs. Note, that for the moment we assume to have a valid
one, a detailed discussion of the config files for shape optimization is given in the [next demo](#documentation-of-the-config-files-for-shape-optimization-problems).

As before, we start the problem by importing everything from FEniCS, and importing cashocs.

    from fenics import *
    import cashocs

Thereafter, we read the config file with the `cashocs.create_config` command

    config = cashocs.create_config('./config.ini')

Next, we have to define the mesh. As the above problem is posed on the unit disc initially, we define this via FEniCS commands (cashocs only has rectangular meshes built
in). This is done via the following code

    meshlevel = 10
    degree = 1
    dim = 2
    mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)

Next up, we define the `Measure` objects, which we need to define the optimization problem. For the volume measure, we can simply invoke

    dx = Measure('dx', mesh)

However, for the surface measure, we need to mark the boundary. This is required since
cashocs distinguishes between three types of boundary: The deformable boundary, the
fixed boundary, and boundaries that can only be deformed perpendicular to a certain
coordinate axis. Most importantly for this example, one of the deformable boundaries
must not be empty, so that we can actually optimize something. In this example,
we investigate the case of a completely deformable boundary, which makes things slightly
easier. We mark this boundary with the marker `1` with the following piece of code

    boundary = CompiledSubDomain('on_boundary')
    boundaries = MeshFunction('size_t', mesh, dim=1)
    boundary.mark(boundaries, 1)
    ds = Measure('ds', mesh, subdomain_data=boundaries)

Note, that all of the alternative ways of marking subdomains or boundaries with
numbers, as explained in [Langtangen and Logg, Solving PDEs in Python](https://doi.org/10.1007/978-3-319-52462-7) also work here. If it is valid for FEniCS, it is also for
cashocs.

After having defined the initial geometry, we define a `FunctionSpace` consisting of
piecewise linear Lagrange elements via

    V = FunctionSpace(mesh, 'CG', 1)
    u = Function(V)
    p = Function(V)

This also defines our state variable \( u \) as `u`, and the adjoint state is given by
`p`. As remarked in [the first demo for optimal control problems](#demo-01-basics), in
classical FEniCS syntax we would use a `TrialFunction` for `u` and a `TestFunction` for
`p`. However, for cashocs this must not be the case. Instead, the state and adjoint
variables have to be `Function` objects.

The right-hand side of the PDE constraint is then defined as

    x = SpatialCoordinate(mesh)
    f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

which allows us to define the weak form of the state equation via

    e = inner(grad(u), grad(p))*dx - f*p*dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

**The optimization problem and its solution**

We are now almost done, the only thing left to do is to define the cost functional

    J = u*dx

and the shape optimization problem

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)

This can then be solved in complete analogy to [Demo 01 : Basics](#demo-01-basics) with
the `solve()` command

    sop.solve()

A picture of the corresponding optimized geometry is given by
![](../demos/documented/shape_optimization/01_shape_poisson/opt_mesh_poisson.png)

> As in [the first optimal control demo](#demo-01-basics) we can specify some keyword
> arguments for the `solve` command. If none are given, then the settings from the
> config file are used, but if some are given, they override the parameters specified
> in the config file. In particular, these arguments are
>
>   - `algorithm` : Specifies which solution algorithm shall be used.
>   - `rtol` : The relative tolerance for the optimization algorithm.
>   - `atol` : The absolute tolerance for the optimization algorithm.
>   - `max_iter` : The maximum amount of iterations that can be carried out.
>
> The choices for these parameters are discussed in detail in the [following demo](#documentation-of-the-config-files-for-shape-optimization-problems).

This concludes the demo, and the corresponding full code can
be found in the file demos/documented/shape_optimization/01_shape_poisson.py
