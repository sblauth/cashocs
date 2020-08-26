"""
Created on 25/08/2020, 14.52

@author: blauths
"""

import numpy as np
import fenics
import warnings



def control_gradient_test(ocp, u=None, h=None):
	"""Taylor test to verify that the computed gradient is correct.

	Parameters
	----------
	ocp : cestrel.OptimalControlProblem
		The underlying optimal control problem, for which the gradient
		of the reduced cost function shall be verified.
	u : list[dolfin.function.function.Function], optional
		The point, at which the gradient shall be verified. If this is None,
		then the current controls of the optimization problem are used.
	h : list[dolfin.function.function.Function], optional
		The direction(s) for the directional (Gateaux) derivative. If this is None,
		one random direction is chosen.

	Returns
	-------
	float
		The convergence order from the Taylor test. If this is (close to) 2, everything works
		as expected.
	"""

	initial_state = []
	for j in range(ocp.control_dim):
		temp = fenics.Function(ocp.form_handler.control_spaces[j])
		temp.vector()[:] = ocp.controls[j].vector()[:]
		initial_state.append(temp)


	if u is None:
		u = []
		for j in range(ocp.control_dim):
			temp = fenics.Function(ocp.form_handler.control_spaces[j])
			temp.vector()[:] = ocp.controls[j].vector()[:]
			u.append(temp)

	assert len(u) == ocp.control_dim, 'Length of u does not match the length of controls of the problem.'

	# check if u and ocp.controls coincide, if yes, make a deepcopy
	ids_u = [fun.id() for fun in u]
	ids_controls = [fun.id() for fun in ocp.controls]
	if ids_u == ids_controls:
		u = []
		for j in range(ocp.control_dim):
			temp = fenics.Function(ocp.form_handler.control_spaces[j])
			temp.vector()[:] = ocp.controls[j].vector()[:]
			u.append(temp)

	if h is None:
		h = []
		for V in ocp.form_handler.control_spaces:
			temp = fenics.Function(V)
			temp.vector()[:] = np.random.rand(V.dim())
			h.append(temp)

	for j in range(ocp.control_dim):
		ocp.controls[j].vector()[:] = u[j].vector()[:]

	# Compute the norm of u for scaling purposes.
	scaling = np.sqrt(ocp.form_handler.scalar_product(ocp.controls, ocp.controls))
	if scaling < 1e-3:
		scaling = 1.0

	ocp.state_problem.has_solution = False
	Ju = ocp.reduced_cost_functional.evaluate()
	ocp.adjoint_problem.has_solution = False
	ocp.gradient_problem.has_solution = False
	grad_Ju = ocp.compute_gradient()
	grad_Ju_h = ocp.form_handler.scalar_product(grad_Ju, h)

	epsilons = [scaling*1e-2 / 2**i for i in range(4)]
	residuals = []

	for eps in epsilons:
		for j in range(ocp.control_dim):
			ocp.controls[j].vector()[:] = u[j].vector()[:] + eps * h[j].vector()[:]
		ocp.state_problem.has_solution = False
		Jv = ocp.reduced_cost_functional.evaluate()

		res = abs(Jv - Ju - eps*grad_Ju_h)
		residuals.append(res)

	if np.min(residuals) < 1e-14:
		warnings.warn('The Taylor remainder is close to 0, results may be inaccurate.')

	rates = compute_convergence_rates(epsilons, residuals)

	for j in range(ocp.control_dim):
		ocp.controls[j].vector()[:] = initial_state[j].vector()[:]

	return np.min(rates)



def shape_gradient_test(sop, h=None):
	"""Taylor test to verify that the computed shape gradient is correct.

	Parameters
	----------
	sop : cestrel.ShapeOptimizationProblem
		The underlying shape optimization problem.
	h : dolfin.function.function.Function, optional
		The direction used to compute the directional derivative. If this is
		None, then a random direction is used (default is None).

	Returns
	-------
	float
		The computed convergence rate. The computed gradient is correct, if
		this quanitity is (about) 2.
	"""

	if h is None:
		h = fenics.Function(sop.shape_form_handler.deformation_space)
		h.vector()[:] = np.random.rand(sop.shape_form_handler.deformation_space.dim())

	transformation = fenics.Function(sop.shape_form_handler.deformation_space)

	sop.state_problem.has_solution = False
	J_curr = sop.reduced_cost_functional.evaluate()
	sop.adjoint_problem.has_solution = False
	sop.shape_gradient_problem.has_solution = False
	shape_grad = sop.compute_shape_gradient()
	shape_derivative_h = sop.shape_form_handler.scalar_product(shape_grad, h)

	box_lower = np.max(sop.mesh_handler.mesh.coordinates())
	box_upper = np.min(sop.mesh_handler.mesh.coordinates())
	length = box_upper - box_lower


	epsilons = [length*1e-4 / 2**i for i in range(4)]
	residuals = []

	for idx, eps in enumerate(epsilons):
		transformation.vector()[:] = eps*h.vector()[:]
		if sop.mesh_handler.move_mesh(transformation):
			sop.state_problem.has_solution = False
			J_pert = sop.reduced_cost_functional.evaluate()

			res = abs(J_pert - J_curr - eps*shape_derivative_h)
			residuals.append(res)
			sop.mesh_handler.revert_transformation()
		else:
			warnings.warn('Deformation did not yield a valid finite element mesh. Results of the test are probably not accurate.')
			residuals.append(float('inf'))

	if np.min(residuals) < 1e-14:
		warnings.warn('The Taylor remainder is close to 0, results may be inaccurate.')

	rates = compute_convergence_rates(epsilons, residuals)

	return np.min(rates)



def compute_convergence_rates(epsilons, residuals, verbose=True):

	rates = []
	for i in range(1, len(epsilons)):
		rates.append(np.log(residuals[i] / residuals[i - 1]) / np.log(epsilons[i] / epsilons[i - 1]))

	if verbose:
		print('Taylor test convergence rate: ' + str(rates))

	return rates
