"""
Created on 11/08/2020, 15.13

@author: blauths
"""

from fenics import *
import caospy
import numpy as np



"""
In this demo we investigate the "mother" optimal control problem 
	
	min J(y, u) = 1/2 * || y - y_d ||_{L^2}^2 + alpha/2 * || u ||_{L^2}^2
	
	s.t. - div ( grad ( y ) ) = u	in \Omega
		 					y = 0	on \partial\Omega
	
		and            u_a <= u <= u_b	in \Omega

using caospy. The parameters for this problem are specified and commented in the parameter file at './config.ini'. 
The difference to the previous problem is in the fact, that we now also consider L^\infty box constraints.
"""

### The initial setup is completely identical to our previous problem
set_log_level(LogLevel.CRITICAL)
config = caospy.create_config('./config.ini')
mesh, subdomains, boundaries, dx, ds, dS = caospy.regular_mesh(50)
V = FunctionSpace(mesh, 'CG', 1)

y = Function(V)
p = Function(V)
u = Function(V)

e = inner(grad(y), grad(p))*dx - u*p*dx

bc1 = DirichletBC(V, Constant(0), boundaries, 1)
bc2 = DirichletBC(V, Constant(0), boundaries, 2)
bc3 = DirichletBC(V, Constant(0), boundaries, 3)
bc4 = DirichletBC(V, Constant(0), boundaries, 4)
bcs = [bc1, bc2, bc3, bc4]

y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)

alpha = 1e-6
J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx

### Define the box constraints. For this example, we use an (affine) linear corridor along the x-axis that bounds the functions.
u_a = interpolate(Expression('50*(x[0]-1)', degree=1), V)
u_b = interpolate(Expression('50*x[0]', degree=1), V)
### Alternatively, we could also consider constant functions, that can be specified directly as floats
# u_a = 0.0
# u_b = float('inf')

### We put them into a list, and into an optional argument of OptimalControlProblem, which is then solved.
control_constraints = [u_a, u_b]
optimization_problem = caospy.OptimalControlProblem(e, bcs, J, y, u, p, config, control_constraints=control_constraints)
optimization_problem.solve()

### Afterwards, we verify that the control constraints are indeed satisfied.
assert np.alltrue(u_a.vector()[:] <= u.vector()[:]) and np.alltrue(u.vector()[:] <= u_b.vector()[:])
### Alternative code for the constant box constraints
# assert np.alltrue(u_a <= u.vector()[:]) and np.alltrue(u.vector()[:] <= u_b)
