Getting started
---------------

Since cashocs is based on FEniCS, most of the user input consists of definining
the objects (such as the state system and cost functional) via UFL forms. If one
has a functioning code for the forward problem and the evaluation of the cost
functional, the necessary modifications to optimize the problem in cashocs
are minimal. Consider, e.g., the following optimization problem

$$ \min J(y, u) = \frac{1}{2} \int_{\Omega} \lvert y - y_d \rvert^2 \text{d}x + \frac{\alpha}{2} \int_\Omega u^2 \text{d}x \\
\text{ subject to }
\begin{aligned}
- \Delta y &= u \quad \text{ in } \Omega, \\
y &= 0 \quad \text{ on } \Gamma.
\end{aligned}
$$

Note, that the entire problem is treated in detail in demo_01.py in the demos folder.

For our purposes, we assume that a mesh for this problem is defined and that a
suitable function space is chosen. This can, e.g., be achieved via

    from fenics import *
    import cashocs

    config = cashocs.create_config('path_to_config')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

The config object which is created from a .ini file is used to determine the
parameters for the optimization algorithms.

To define the state problem, we then define a state variable y, an adjoint variable
p and a control variable u, and write the PDE as a weak form

    y = Function(V)
    p = Function(V)
    u = Function(V)
    e = inner(grad(y), grad(p)) - u*p*dx
    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])

Finally, we have to define the cost functional and the optimization problem

    y_d = Expression('sin(2*pi * x[0] * sin(2*pi*x[1]))', degree=1)
    alpha = 1e-6
    J = 1/2*(y - y_d) * (y - y_d) * dx + alpha/2*u*u*dx
    opt_problem = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    opt_problem.solve()

The only major difference between cashocs and fenics code is that one has to
use Function objects for states and adjoints, and that Trial- and TestFunctions
are not needed to define the state equation. Other than that, the syntax would
also be valid with fenics.

For a detailed discussion of the features of cashocs and its usage we refer to the [demos](#demos).


Demos
-----
The documentation of the demos can be found <a target="_blank" rel="noopener noreferrer" href="./doc_demos.html">here</a>.

Note, that cashocs was also used to obtain the numerical results for my preprints
[Blauth, Leithäuser, and Pinnau, Optimal Control of the Sabatier Process in Microchannel Reactors](url) and [Blauth, Nonlinear Conjugate Gradient Methods for PDE Constrained Shape Optimization based on Steklov-Poincare Type Metrics](url)


Command line interface for mesh conversion
------------------------------------------

cashocs includes a command line interface for converting gmsh mesh files to
xdmf ones, which can be read very easily into fenics. The corresponding command
for the conversion (after having generated a mesh file 'in.msh' with gmsh)
is given by

    cashocs-convert in.msh out.xdmf

This also create .xdmf files for subdomains and boundaries in case they are tagged
in gmsh as Physical quantities.


License
-------

CASHOCS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CASHOCS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.


Contact / About
---------------

I'm Sebastian Blauth, a PhD student at Fraunhofer ITWM and TU Kaiserslautern,
and I developed this project as part of my work. If you have any questions /
suggestions / feedback, etc., you can contact me via
[sebastian.blauth@itwm.fraunhofer.de](mailto:sebastian.blauth@itwm.fraunhofer.de).



Copyright (C) 2020 Sebastian Blauth
