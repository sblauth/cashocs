"""
Created on 11/08/2020, 15.44

@author: blauths
"""

from fenics import *
import adpack
import numpy as np



"""
In this demo, we investigate the case of handling multiple state equations that are one-way coupled, 
e.g., for time dependent problems or even parameter identification, where there is no coupling at all.
In particular, we consider the following problem
	
	min J((y, z), (u, v)) = 1/2 * || y - y_d ||_{L^2}^2 + 1/2 * || z - z_d ||_{L^2}^2
							+ alpha/2 * || u ||_{L^2}^2 + beta/2 * || v ||_{L^2}^2
	
	s.t. - div ( grad ( y ) ) = u		in \Omega
		 - div ( grad ( z ) ) = y + v	in \Omega
		 					y = 0		on \partial\Omega
		 					z = 0		on \partial\Omega
	
		and            u_a <= u <= u_b	in \Omega
					   v_a <= v <= v_b 	in \Omega

The parameters for this problem are specified and commented in the parameter file at './config.ini'. 
"""

### The initial setup is completely identical to our previous problem
set_log_level(LogLevel.CRITICAL)
config = adpack.create_config('./config.ini')
mesh, subdomains, boundaries, dx, ds, dS = adpack.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

### Now, we define two state, adjoint, and control variables, where p and q are the adjoint states corresponding to y and z
y = Function(V)
z = Function(V)
p = Function(V)
q = Function(V)
u = Function(V)
v = Function(V)

### For better readability, we collect them to lists now. NOTE: The order of the states and adjoints has to match!
state_variables = [y, z]
adjoint_variables = [p, q]
control_variables = [u, v]

# We define both state equations seperately
e1 = inner(grad(y), grad(p))*dx - u*p*dx
e2 = inner(grad(z), grad(q))*dx - (y + v)*q*dx
### and collect them also to a list. NOTE: the order is again important
e = [e1, e2]

### Next, we define th Dirichlet boundary conditions (which are the same for both state variables)
bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs1 = [bc1, bc2, bc3, bc4]
bcs2 = [bc1, bc2, bc3, bc4]
### and collect them to a list, too. Again, the order is important
bcs = [bcs1, bcs2]

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
z_d = Expression('sin(4*pi*x[0])*sin(4*pi*x[1])', degree=1)

alpha = 1e-6
beta = 1e-4
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5)*(z - z_d)*(z - z_d)*dx \
	+ Constant(0.5*alpha)*u*u*dx + Constant(0.5*beta)*v*v*dx

### Define the box constraints, similarly to the previous example
u_a = interpolate(Expression('50*(x[0]-1)', degree=1), V)
u_b = interpolate(Expression('50*x[0]', degree=1), V)
v_a = 0.0
v_b = float('inf')

### We put them into a list, and into an optional argument of OptimalControlProblem, which is then solved.
control_constraints = [[u_a, u_b], [v_a, v_b]]
optimization_problem = adpack.OptimalControlProblem(e, bcs, J, state_variables, control_variables, adjoint_variables, config, control_constraints=control_constraints)
optimization_problem.solve()

