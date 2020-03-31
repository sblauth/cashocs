"""
Created on 24/02/2020, 16.48

@author: blauths
"""

import fenics
import numpy as np



class HessianProblem:
	
	def __init__(self, form_handler, gradient_problem, control_constraints):
		"""The class that manages the computations for the Hessian in the truncated Newton method
		
		Parameters
		----------
		form_handler : adpack.forms.FormHandler
			the FormHandler object for the optimization problem
		gradient_problem : adpack.pde_problems.gradient_problem.GradientProblem
			the GradientProblem object (we need the gradient for the computation of the Hessian)
		"""
		
		self.form_handler = form_handler
		self.gradient_problem = gradient_problem
		self.control_constraints = control_constraints
		
		self.config = self.form_handler.config
		self.gradients = self.gradient_problem.gradients
		
		self.inner_newton = self.config.get('OptimizationRoutine', 'inner_newton')
		self.max_it_inner_newton = self.config.getint('OptimizationRoutine', 'max_it_inner_newton')
		self.inner_newton_tolerance = self.config.getfloat('OptimizationRoutine', 'inner_newton_tolerance')
		
		self.test_directions = self.form_handler.test_directions
		self.residual = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.delta_control = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.state_dim = self.form_handler.state_dim
		self.control_dim = self.form_handler.control_dim

		self.p = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.q = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.q_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.r_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.inactive_part = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.active_part = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.controls = self.form_handler.controls

		self.rtol = self.config.getfloat('StateEquation', 'picard_rtol')
		self.atol = self.config.getfloat('StateEquation', 'picard_atol')
		self.maxiter = self.config.getint('StateEquation', 'picard_iter')
		self.picard_verbose = self.config.getboolean('StateEquation', 'picard_verbose')
		
		self.no_sensitivity_solves = 0
		
	
	
	def hessian_application(self):
		"""Returns the application of J''(u)[h] where h = self.test_direction
		
		This is needed in the truncated Newton method where we solve the system
			J''(u) du = - J'(u)
		via iterative methods (cg, minres, cr)
		
		Returns
		-------
		self.form_handler.hessian_action : List[dolfin.function.function.Function]
			the generic function that saves the result of J''(u)[h]

		"""
		
		self.states_prime = self.form_handler.states_prime
		self.adjoints_prime = self.form_handler.adjoints_prime
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		if not self.config.getboolean('StateEquation', 'picard_iteration'):

			for i in range(self.state_dim):
				fenics.solve(self.form_handler.sensitivity_eqs_lhs[i]==self.form_handler.sensitivity_eqs_rhs[i], self.states_prime[i], self.bcs_list_ad[i])

			for i in range(self.state_dim):
				fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1-i]==self.form_handler.w_1[-1-i], self.adjoints_prime[-1-i], self.bcs_list_ad[-1-i])


		else:
			for i in range(self.maxiter + 1):
				res = 0.0
				for j in range(self.form_handler.state_dim):
					res_j = fenics.assemble(self.form_handler.sensitivity_eqs_picard[j])
					[bc.apply(res_j) for bc in self.form_handler.bcs_list_ad[j]]
					res += pow(res_j.norm('l2'), 2)

				if res==0:
					break

				res = np.sqrt(res)

				if i==0:
					res_0 = res

				if self.picard_verbose:
					print('Picard Sensitivity 1 Iteration ' + str(i) + ': ||res|| (abs): ' + format(res, '.3e') + '   ||res|| (rel): ' + format(res/res_0, '.3e'))

				if res/res_0 < self.rtol or res < self.atol:
					break

				if i==self.maxiter:
					raise SystemExit('Failed to solve the Picard Iteration')

				for j in range(self.form_handler.state_dim):
					fenics.solve(self.form_handler.sensitivity_eqs_lhs[j]==self.form_handler.sensitivity_eqs_rhs[j], self.states_prime[j], self.bcs_list_ad[j])


			if self.picard_verbose:
				print('')

			for i in range(self.maxiter + 1):
				res = 0.0
				for j in range(self.form_handler.state_dim):
					res_j = fenics.assemble(self.form_handler.adjoint_sensitivity_eqs_picard[j])
					[bc.apply(res_j) for bc in self.form_handler.bcs_list_ad[j]]
					res += pow(res_j.norm('l2'), 2)

				if res==0:
					break

				res = np.sqrt(res)

				if i==0:
					res_0 = res

				if self.picard_verbose:
					print('Picard Sensitivity 2 Iteration ' + str(i) + ': ||res|| (abs): ' + format(res, '.3e') + '   ||res|| (rel): ' + format(res/res_0, '.3e'))

				if res/res_0 < self.rtol or res < self.atol:
					break

				if i==self.maxiter:
					raise SystemExit('Failed to solve the Picard Iteration')

				for j in range(self.form_handler.state_dim):
					fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1-j]==self.form_handler.w_1[-1-j], self.adjoints_prime[-1 - j], self.bcs_list_ad[-1 - j])


			if self.picard_verbose:
				print('')


		for i in range(self.control_dim):
			A = fenics.assemble(self.form_handler.hessian_lhs[i], keep_diagonal=True)
			A.ident_zeros()
			b = fenics.assemble(self.form_handler.hessian_rhs[i])
			fenics.solve(A, self.form_handler.hessian_actions[i].vector(), b)
		
		self.no_sensitivity_solves += 2
		
		return self.form_handler.hessian_actions


	
	def application_simplified(self, x):
		"""A simplified version of the application of the Hessian.
		
		Computes J''(u)[x], where x is the input vector (see self.hessian_application for more details)
		
		Parameters
		----------
		x : List[dolfin.function.function.Function]
			a function to which we want to apply the Hessian to

		Returns
		-------
		self.hessian_actions : List[dolfin.function.function.Function]
			the generic function that saves the result of J''(u)[h]

		"""
		
		for i in range(self.control_dim):
			self.test_directions[i].vector()[:] = x[i].vector()[:]
		
		self.hessian_actions = self.hessian_application()
		
		return self.hessian_actions

	
	
	def newton_solve(self):
		"""Solves the truncated Newton problem using an iterative method (cg, minres or cr)
		
		Returns
		-------
		self.delta_control : List[dolfin.function.function.Function]
			the Newton increment

		"""

		self.form_handler.compute_active_sets()
		
		for j in range(self.control_dim):
			self.delta_control[j].vector()[:] = 0.0
		self.gradient_problem.solve()
		
		### CG method
		if self.inner_newton == 'cg':
			
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
				self.p[j].vector()[:] = self.residual[j].vector()[:]
			
			self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
			self.eps_0 = np.sqrt(self.res_norm_squared)
			
			for i in range(self.max_it_inner_newton):
				self.form_handler.project_inactive(self.p, self.inactive_part)
				self.hessian_actions = self.application_simplified(self.inactive_part)
				self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
				self.form_handler.project_active(self.p, self.active_part)

				for j in range(self.control_dim):
					self.q[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]
				
				self.eps = np.sqrt(self.res_norm_squared)
				# print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance:
					break
				
				self.alpha = self.res_norm_squared / self.form_handler.scalar_product(self.p, self.q)
				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p[j].vector()[:]
					self.residual[j].vector()[:] -= self.alpha*self.q[j].vector()[:]
				
				self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
				self.beta = self.res_norm_squared / pow(self.eps, 2)
				
				for j in range(self.control_dim):
					self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta*self.p[j].vector()[:]


		elif self.inner_newton == 'minres':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
				self.p_prev[j].vector()[:] = self.residual[j].vector()[:]

			self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.form_handler.project_inactive(self.p_prev, self.inactive_part)
			self.hessian_actions = self.application_simplified(self.inactive_part)
			self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
			self.form_handler.project_active(self.p_prev, self.active_part)

			for j in range(self.control_dim):
				self.s_prev[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]

			for i in range(self.max_it_inner_newton):
				self.alpha = self.form_handler.scalar_product(self.residual, self.s_prev) / self.form_handler.scalar_product(self.s_prev, self.s_prev)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p_prev[j].vector()[:]
					self.residual[j].vector()[:] -= self.alpha*self.s_prev[j].vector()[:]

				self.eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
				# print('Eps (mr): ' + str(self.eps / self.eps_0) + ' (relative)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
					break

				for j in range(self.control_dim):
					self.p[j].vector()[:] = self.s_prev[j].vector()[:]

				self.form_handler.project_inactive(self.s_prev, self.inactive_part)
				self.hessian_actions = self.application_simplified(self.inactive_part)
				self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
				self.form_handler.project_active(self.s_prev, self.active_part)

				for j in range(self.control_dim):
					self.s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]

				self.beta_prev = self.form_handler.scalar_product(self.s, self.s_prev) / self.form_handler.scalar_product(self.s_prev, self.s_prev)
				if i==0:
					self.beta_pprev = 0.0
				else:
					self.beta_pprev = self.form_handler.scalar_product(self.s, self.s_pprev) / self.form_handler.scalar_product(self.s_pprev, self.s_pprev)

				for j in range(self.control_dim):
					self.p[j].vector()[:] -= self.beta_prev*self.p_prev[j].vector()[:] + self.beta_pprev*self.p_pprev[j].vector()[:]
					self.s[j].vector()[:] -= self.beta_prev*self.s_prev[j].vector()[:] + self.beta_pprev*self.s_pprev[j].vector()[:]

					self.p_pprev[j].vector()[:] = self.p_prev[j].vector()[:]
					self.s_pprev[j].vector()[:] = self.s_prev[j].vector()[:]

					self.p_prev[j].vector()[:] = self.p[j].vector()[:]
					self.s_prev[j].vector()[:] = self.s[j].vector()[:]


		elif self.inner_newton == 'cr':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
				self.p[j].vector()[:] = self.residual[j].vector()[:]

			self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.form_handler.project_inactive(self.residual, self.inactive_part)
			self.hessian_actions = self.application_simplified(self.inactive_part)
			self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
			self.form_handler.project_active(self.residual, self.active_part)

			for j in range(self.control_dim):
				self.s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]
				self.q[j].vector()[:] = self.s[j].vector()[:]

			self.rAr = self.form_handler.scalar_product(self.residual, self.s)

			for i in range(self.max_it_inner_newton):
				self.alpha = self.rAr / self.form_handler.scalar_product(self.q, self.q)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p[j].vector()[:]
					self.r_prev[j].vector()[:] = self.residual[j].vector()[:]
					self.residual[j].vector()[:] -= self.alpha*self.q[j].vector()[:]

				self.eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
				# print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
					break

				self.form_handler.project_inactive(self.residual, self.inactive_part)
				self.hessian_actions = self.application_simplified(self.inactive_part)
				self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
				self.form_handler.project_active(self.residual, self.active_part)
				for j in range(self.control_dim):
					self.s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]

				self.rAr_new = self.form_handler.scalar_product(self.residual, self.s)
				self.beta = self.rAr_new / self.rAr

				for j in range(self.control_dim):
					self.q_prev[j].vector()[:] = self.q[j].vector()[:]
					self.q[j].vector()[:] = self.s[j].vector()[:] + self.beta*self.q[j].vector()[:]
					self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta*self.p[j].vector()[:]


				self.rAr = self.rAr_new

		else:
			raise SystemExit('OptimizationRoutine.inner_newton needs to be one of cg, minres or cr.')
		
		return self.delta_control
