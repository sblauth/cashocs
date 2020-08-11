"""
Created on 15/01/2020, 17.20

@author: blauths
"""

import fenics



def NewtonSolver(F, u, bcs, rtol=1e-10, atol=1e-10, max_iter=25, convergence_type='both', norm_type='l2', damped=True, verbose=True):
	"""A custom NewtonSolver class that includes damping (based on a monotonicity test) and fine tuning possibilities.

	The solver will stop either after the maximum number of iterations have been performed or if the termination criterion is satisfied, i.e.,
		if ||F_current|| <= rtol * ||F_initial|| 	if convergence_type is 'rel',
		if ||F_current|| <= atol 					if convergence_type is 'abs',
		or when either of the above is satisfied 	if convergence_type is 'both'.
	The corresponding norm is specified via 'norm_type'.

	Parameters
	----------
	F : ufl.form.Form
		The variational form of the nonlinear problem that shall be solved by Newton's method

	u : dolfin.function.function.Function
		The sought solution / initial guess (the method will also return this)

	bcs : list[dolfin.fem.dirichletbc.DirichletBC]
		A list of DirichletBCs for the nonlinear variational problem

	rtol : float
		Relative tolerance of the solver if convergence_type is either 'both' or 'rel'

	atol : float
		Absolute tolerance of the solver if convergence_type is either 'both' or 'abs'

	max_iter : int
		Maximum number of iterations carried out by the method

	convergence_type : str
		One of 'both', 'rel', or 'abs'

	norm_type : str
		One of 'l2', 'linf'

	damped : bool
		Either true or false, if true then uses a damping strategy, if not, uses Newton-Raphson Iteration

	verbose : bool
		Either true or false, gives updates about the iteration, if true

	Returns
	-------
	u : dolfin.function.function.Function
		The solution of the nonlinear variational problem, if converged
	"""

	assert convergence_type in ['rel', 'abs', 'both'], 'Newton Solver convergence_type is neither rel nor abs nor both'
	assert norm_type in ['l2', 'linf'], 'Newton Solver norm_type is neither l2 nor linf'
	
	fenics.PETScOptions.set('mat_mumps_icntl_24', 1)
	
	dF = fenics.derivative(F, u)
	V = u.function_space()
	
	du = fenics.Function(V)
	ddu = fenics.Function(V)
	u_save = fenics.Function(V)
	
	iterations = 0
	
	[bc.apply(u.vector()) for bc in bcs]
	# copy the boundary conditions and homogenize them
	bcs_hom = [fenics.DirichletBC(bc) for bc in bcs]
	[bc.homogenize() for bc in bcs_hom]
	
	A, residuum = fenics.assemble_system(dF, -F, bcs_hom)
	
	res_0 = residuum.norm(norm_type)
	res = res_0
	if verbose:
		print('Newton Iteration ' + format(iterations, '2d') + ' - residuum (abs):  ' + format(res, '.3e') + ' (tol = ' + format(atol, '.3e') + ')    residuum (rel): '
		  		+ format(res/res_0, '.3e') + ' (tol = ' + format(rtol, '.3e') + ')')
	
	if convergence_type == 'abs':
		tol = atol
	elif convergence_type == 'rel':
		tol = rtol*res_0
	else:
		tol = rtol*res_0 + atol
	
	while res > tol and iterations < max_iter:
		iterations += 1
		lmbd = 1.0
		breakdown = False
		u_save.vector()[:] = u.vector()[:]
		
		fenics.solve(A, du.vector(), residuum, 'mumps')
		# fenics.solve(A, du.vector(), residuum, 'umfpack')
		
		if damped:
			while True:
				u.vector()[:] += lmbd*du.vector()[:]
				_, b = fenics.assemble_system(dF, -F, bcs_hom)
				
				fenics.solve(A, ddu.vector(), b, 'mumps')
				# fenics.solve(A, ddu.vector(), b, 'umfpack')
				
				if ddu.vector().norm(norm_type)/du.vector().norm(norm_type) <= 1:
					break
				else:
					u.vector()[:] = u_save.vector()[:]
					lmbd /= 2
				
				if lmbd < 1e-6:
					breakdown = True
					break
		
		else:
			u.vector()[:] += du.vector()[:]
		
		if breakdown:
			raise SystemExit('Damped Newton Solver Breakdown')

		if iterations == max_iter:
			raise SystemExit('Newton Solver exceeded maximum number of iterations.')
		
		A, residuum = fenics.assemble_system(dF, -F, bcs_hom)
		[bc.apply(residuum) for bc in bcs_hom]
		
		res = residuum.norm(norm_type)
		if verbose:
			print('Newton Iteration ' + format(iterations, '2d') + ' - residuum (abs):  ' + format(res, '.3e') + ' (tol = ' + format(atol, '.3e') + ')    residuum (rel): '
				  + format(res/res_0, '.3e') + ' (tol = ' + format(rtol, '.3e') + ')')
		
		if res < tol:
			if verbose:
				print('')
				print('Newton Solver converged after ' + str(iterations) + ' iterations.')
			break
	
	return u
