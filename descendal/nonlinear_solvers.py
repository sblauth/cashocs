"""Custom solvers for nonlinear equations

This module has custom solvers for nonlinear PDEs, including
a damped Newton method. Other methods might follow in the future.
"""

import fenics
from petsc4py import PETSc



def damped_newton_solve(F, u, bcs, rtol=1e-10, atol=1e-10, max_iter=50, convergence_type='combined', norm_type='l2',
						damped=True, verbose=True, ksp=None):
	r"""A damped Newton method for solving nonlinear equations.

	The Newton method is based on the natural monotonicity test from Deuflhard.
	It also allows fine tuning via a direct interface, and absolute, relative,
	and combined stopping criteria. Can also be used to specify the solver for
	the inner (linear) subproblems via petsc ksps.

	The method terminates after max_iter iterations, or if a termination criterion is
	satisfied. These criteria are given by
	$$ \lvert\lvert F_{k} \rvert\rvert \leq \text{rtol} \lvert\lvert F_0 \rvert\rvert \quad \text{ if convergence_type is 'rel'} \\
	\lvert\lvert F_{k} \rvert\rvert \leq \text{atol} \quad \text{ if convergence_type is 'abs'} \\
	\lvert\lvert F_{k} \rvert\rvert \leq \text{atol} + \text{rtol} \lvert\lvert F_0 \rvert\rvert \quad \text{ if convergence_type is 'combined'}
	$$

	The norm chosen for this termination criterion is specified via norm_type.

	Parameters
	----------
	F : ufl.form.Form
		The variational form of the nonlinear problem to be solved by Newton's method.
	u : dolfin.function.function.Function
		The sought solution / initial guess (the method will also return this). It is
		not assumed that the initial guess satisfies the Dirichlet boundary conditions,
		these are applied automatically.
	bcs : list[dolfin.fem.dirichletbc.DirichletBC]
		A list of DirichletBCs for the nonlinear variational problem.
	rtol : float, optional
		Relative tolerance of the solver if convergence_type is either 'combined' or 'rel'
		(default is rtol = 1e-10).
	atol : float, optional
		Absolute tolerance of the solver if convergence_type is either 'combined' or 'abs'
		(default is atol = 1e-10).
	max_iter : int, optional
		Maximum number of iterations carried out by the method
		(default is max_iter = 50).
	convergence_type : str, optional
		Determines the type of stopping criterion that is used.
		One of 'combined', 'rel', or 'abs' (default is 'combined').
	norm_type : str, optional
		Determines which norm is used in the stopping criterion.
		One of 'l2', 'linf' (default is 'l2').
	damped : bool, optional
		If true, then a damping strategy is used. If false, the classical
		Newton-Raphson iteration (without damping) is used (default is True).
	verbose : bool, optional
		If true, prints status of the iteration to the console (default
		is true).
	ksp : petsc4py.PETSc.KSP, optional
		The PETSc ksp object used to solve the inner (linear) problem
		if this is None it uses the direct solver MUMPS (default is
		None)

	Returns
	-------
	dolfin.function.function.Function
		The solution of the nonlinear variational problem, if converged.
		This overrides the input function u.
	"""

	assert convergence_type in ['rel', 'abs', 'combined'], \
		'Input convergence_type has to be one of \'rel\', \'abs\', or \'combined\''
	assert norm_type in ['l2', 'linf'], \
		'Input norm_type has to be one of \'l2\' or \'linf\''

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
