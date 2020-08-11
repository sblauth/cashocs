"""
Created on 11/08/2020, 14.35

@author: blauths
"""

from fenics import *
import adpack



"""
In this demo we investigate the "mother" optimal control problem 
	
	min J(y, u) = 1/2 * || y - y_d ||_{L^2}^2 + alpha/2 * || u ||_{L^2}^2
	
	s.t. - div ( grad ( y ) ) = u	in \Omega
		 					y = 0	on \partial\Omega

using adpack. The parameters for this problem are specified and commented in the parameter file at './config.ini'. 
"""

### Supress FEniCS default verbose output
set_log_level(LogLevel.CRITICAL)

### load the config file
config = adpack.create_config('./config.ini')

### Generate the mesh (via gmsh)
# mesh, subdomains, boundaries, dx, ds, dS = adpack.MeshGen('../mesh/mesh.xdmf')
### Alternatively: Generate the mesh (via built-ins)
mesh, subdomains, boundaries, dx, ds, dS = adpack.regular_mesh(50)

### Set up a function space with piecewise linear Lagrangian elements
V = FunctionSpace(mesh, 'CG', 1)

### Define state variable y and adjoint variable p (NOTE: always use Function for this, no Trial- or TestFunction)
y = Function(V)
p = Function(V)

### Define control variable u
u = Function(V)

### Define the weak form of the state equation, by "testing with the adjoint variable"
e = inner(grad(y), grad(p))*dx - u*p*dx

### Define the boundary conditions
bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

### Define the desired state y_d
y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)

### Define the cost functional J
alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

### Set up the optimization problem and solve it
optimization_problem = adpack.OptimalControlProblem(e, bcs, J, y, u, p, config)
optimization_problem.solve()
