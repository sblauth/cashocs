"""
Created on 01.04.20, 15:15

@author: sebastian
"""

import fenics
import numpy as np



class UnconstrainedHessianProblem:

	def __init__(self, form_handler, gradient_problem):

		self.form_handler = form_handler
		self.gradient_problem = gradient_problem

		self.config = self.form_handler.config
		self.gradients = self.gradient_problem.gradients
		self.reduced_gradient = [fenics.Function(self.form_handler.control_spaces[j]) for j in range(len(self.gradients))]

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

		self.states_prime = self.form_handler.states_prime
		self.adjoints_prime = self.form_handler.adjoints_prime
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		if not self.config.getboolean('StateEquation', 'picard_iteration'):

			for i in range(self.state_dim):
				fenics.solve(self.form_handler.sensitivity_eqs_lhs[i]==self.form_handler.sensitivity_eqs_rhs[i], self.states_prime[i], self.bcs_list_ad[i])

			for i in range(self.state_dim):
				fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1 - i]==self.form_handler.w_1[-1 - i], self.adjoints_prime[-1 - i], self.bcs_list_ad[-1 - i])


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
					fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1 - j]==self.form_handler.w_1[-1 - j], self.adjoints_prime[-1 - j], self.bcs_list_ad[-1 - j])

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

		for i in range(self.control_dim):
			self.test_directions[i].vector()[:] = x[i].vector()[:]

		self.hessian_actions = self.hessian_application()

		return self.hessian_actions

	def newton_solve(self, idx_active):

		# self.form_handler.compute_active_sets()

		for j in range(self.control_dim):
			self.delta_control[j].vector()[:] = 0.0
		self.gradient_problem.solve()

		for j in range(self.control_dim):
			self.reduced_gradient[j].vector()[:] = self.gradients[j].vector()[:]
			self.reduced_gradient[j].vector()[idx_active[j]] = 0.0

		### CG method
		if self.inner_newton=='cg':

			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
				self.p[j].vector()[:] = self.residual[j].vector()[:]

			self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
			self.eps_0 = np.sqrt(self.res_norm_squared)

			for i in range(self.max_it_inner_newton):

				for j in range(self.control_dim):
					self.p[j].vector()[idx_active[j]] = 0.0

				self.hessian_actions = self.application_simplified(self.p)

				for j in range(self.control_dim):
					self.q[j].vector()[:] = self.hessian_actions[j].vector()[:]
					self.q[j].vector()[idx_active[j]] = 0.0

				self.eps = np.sqrt(self.res_norm_squared)
				# print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
				if self.eps/self.eps_0 < self.inner_newton_tolerance:
					break

				self.alpha = self.res_norm_squared/self.form_handler.scalar_product(self.p, self.q)
				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p[j].vector()[:]
					self.residual[j].vector()[:] -= self.alpha*self.q[j].vector()[:]

				self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
				self.beta = self.res_norm_squared/pow(self.eps, 2)

				for j in range(self.control_dim):
					self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta*self.p[j].vector()[:]


		elif self.inner_newton == 'minres':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
				self.p_prev[j].vector()[:] = self.residual[j].vector()[:]

			self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.hessian_actions = self.application_simplified(self.p_prev)

			for j in range(self.control_dim):
				self.s_prev[j].vector()[:] = self.hessian_actions[j].vector()[:]
				self.s_prev[j].vector()[idx_active[j]] = 0.0

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

				self.hessian_actions = self.application_simplified(self.s_prev)

				for j in range(self.control_dim):
					self.s[j].vector()[:] = self.hessian_actions[j].vector()[:]
					self.s[j].vector()[idx_active[j]] = 0.0

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


		elif self.inner_newton=='cr':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
				self.p[j].vector()[:] = self.residual[j].vector()[:]

			self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.hessian_actions = self.application_simplified(self.residual)

			for j in range(self.control_dim):
				self.s[j].vector()[:] = self.hessian_actions[j].vector()[:]
				self.s[j].vector()[idx_active[j]] = 0.0
				self.q[j].vector()[:] = self.s[j].vector()[:]

			self.rAr = self.form_handler.scalar_product(self.residual, self.s)

			for i in range(self.max_it_inner_newton):
				self.alpha = self.rAr/self.form_handler.scalar_product(self.q, self.q)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p[j].vector()[:]
					self.r_prev[j].vector()[:] = self.residual[j].vector()[:]
					self.residual[j].vector()[:] -= self.alpha*self.q[j].vector()[:]

				self.eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
				# print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
				if self.eps/self.eps_0 < self.inner_newton_tolerance or i==self.max_it_inner_newton - 1:
					break

				self.hessian_actions = self.application_simplified(self.residual)

				for j in range(self.control_dim):
					self.s[j].vector()[:] = self.hessian_actions[j].vector()[:]
					self.s[j].vector()[idx_active[j]] = 0.0

				self.rAr_new = self.form_handler.scalar_product(self.residual, self.s)
				self.beta = self.rAr_new/self.rAr

				for j in range(self.control_dim):
					self.q_prev[j].vector()[:] = self.q[j].vector()[:]
					self.q[j].vector()[:] = self.s[j].vector()[:] + self.beta*self.q[j].vector()[:]
					self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta*self.p[j].vector()[:]

				self.rAr = self.rAr_new

		else:
			raise SystemExit('OptimizationRoutine.inner_newton needs to be one of cg, minres or cr.')

		return self.delta_control
