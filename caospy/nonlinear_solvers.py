"""Custom solvers for Nonlinear PDEs

At the moment only includes a damped Newton
method, others might follow in the future.
"""

import fenics
from petsc4py import PETSc



def newton_solve(F, u, bcs, rtol=1e-10, atol=1e-10, max_iter=25, convergence_type='both', norm_type='l2', damped=True, verbose=True, ksp=None):
	"""Damped Newton method for solving nonlinear PDEs.

	A custom newton_solve class that includes damping
	(based on a monotonicity test) and several fine tuning possibilities.

	The solver will stop either after the maximum number of iterations
	have been performed or if the termination criterion is satisfied, i.e.,

		if ||F_current|| <= rtol * ||F_initial|| 	if convergence_type is 'rel',
		if ||F_current|| <= atol 					if convergence_type is 'abs',
		or when either of the above is satisfied 	if convergence_type is 'both'.

	The corresponding norm is specified via 'norm_type'.

	Parameters
	----------
	F : ufl.form.Form
		The variational form of the nonlinear problem that shall be
		solved by Newton's method
	u : dolfin.function.function.Function
		The sought solution / initial guess (the method will also return this)
	bcs : list[dolfin.fem.dirichletbc.DirichletBC]
		A list of DirichletBCs for the nonlinear variational problem
	rtol : float, optional
		Relative tolerance of the solver if convergence_type is either 'both' or 'rel'
		(default is rtol = 1e-10)
	atol : float, optional
		Absolute tolerance of the solver if convergence_type is either 'both' or 'abs'
		(default is atol = 1e-10)
	max_iter : int, optional
		Maximum number of iterations carried out by the method
		(default is max_iter = 25)
	convergence_type : str, optional
		One of 'both', 'rel', or 'abs' (default is 'both')
	norm_type : str, optional
		One of 'l2', 'linf' (default is 'l2')
	damped : bool, optional
		Either true or false, if true then uses a damping strategy, if not,
		uses Newton-Raphson iteration (default is True)
	verbose : bool, optional
		Either true or false, gives updates about the iteration, if true
		(default is true)
	ksp : petsc4py.PETSc.KSP, optional
		the PETSc ksp object used to solve the inner (linear) problem
		if this is None (the default) it uses the direct solver MUMPS

	Returns
	-------
	u : dolfin.function.function.Function
		The solution of the nonlinear variational problem, if converged
	"""

	assert convergence_type in ['rel', 'abs', 'both'], \
		'Newton Solver convergence_type is neither rel nor abs nor both'
	assert norm_type in ['l2', 'linf'], \
		'Newton Solver norm_type is neither l2 nor linf'

	if ksp is None:
		opts = fenics.PETScOptions
		opts.clear()
		opts.set('ksp_type', 'preonly')
		opts.set('pc_type', 'lu')
		opts.set('pc_factor_mat_solver_type', 'mumps')
		opts.set('mat_mumps_icntl_24', 1)

		ksp = PETSc.KSP().create()
		ksp.setFromOptions()
	
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
	
	A, residuum = fenics.assemble_system(dF, -F, bcs_hom, keep_diagonal=True)
	A.ident_zeros()
	A = fenics.as_backend_type(A).mat()
	b = fenics.as_backend_type(residuum).vec()
	
	res_0 = residuum.norm(norm_type)
	res = res_0
	if verbose:
		print('Newton Iteration ' + format(iterations, '2d') + ' - residuum (abs):  '
			  + format(res, '.3e') + ' (tol = ' + format(atol, '.3e') + ')    residuum (rel): '
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

		ksp.setOperators(A)
		ksp.solve(b, du.vector().vec())

		if damped:
			while True:
				u.vector()[:] += lmbd*du.vector()[:]
				_, b = fenics.assemble_system(dF, -F, bcs_hom)
				b = fenics.as_backend_type(b).vec()
				ksp.solve(b, ddu.vector().vec())
				
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
		
		A, residuum = fenics.assemble_system(dF, -F, bcs_hom, keep_diagonal=True)
		A.ident_zeros()
		A = fenics.as_backend_type(A).mat()
		b = fenics.as_backend_type(residuum).vec()

		[bc.apply(residuum) for bc in bcs_hom]
		
		res = residuum.norm(norm_type)
		if verbose:
			print('Newton Iteration ' + format(iterations, '2d') + ' - residuum (abs):  '
				  + format(res, '.3e') + ' (tol = ' + format(atol, '.3e') + ')    residuum (rel): '
				  + format(res/res_0, '.3e') + ' (tol = ' + format(rtol, '.3e') + ')')
		
		if res < tol:
			if verbose:
				print('')
				print('Newton Solver converged after ' + str(iterations) + ' iterations.')
			break
	
	return u
