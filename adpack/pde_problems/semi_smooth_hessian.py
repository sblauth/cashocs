"""
Created on 06/03/2020, 10.22

@author: blauths
"""

import fenics
import numpy as np
from ..helpers import summ



class SemiSmoothHessianProblem:

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
		self.residual1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.residual2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.delta_control = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.delta_mu = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.state_dim = self.form_handler.state_dim
		self.control_dim = self.form_handler.control_dim

		self.p1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.q = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.mu = [fenics.Function(V) for V in self.form_handler.control_spaces]
		for j in range(self.control_dim):
			self.mu[j].vector()[:] = 0
		self.temp = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp_storage = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.cc = 1e-4

		self.inactive_part = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.active_part = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.controls = self.form_handler.controls

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

		for i in range(self.state_dim):
			fenics.solve(self.form_handler.sensitivity_eqs_lhs[i]==self.form_handler.sensitivity_eqs_rhs[i], self.states_prime[i], self.bcs_list_ad[i])

		for i in range(self.state_dim):
			fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1-i]==self.form_handler.w_1[-1-i], self.adjoints_prime[-1-i], self.bcs_list_ad[-1-i])

		for i in range(self.control_dim):
			A = fenics.assemble(self.form_handler.hessian_lhs[i], keep_diagonal=True)
			A.ident_zeros()
			b = fenics.assemble(self.form_handler.hessian_rhs[i])
			fenics.solve(A, self.form_handler.hessian_actions[i].vector(), b)

		self.no_sensitivity_solves += 2

		return self.form_handler.hessian_actions



	def hessian_application_simplified(self, x):
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



	def compute_sets(self):

		self.idx_active_lower = [(self.mu[j].vector()[:] + self.cc*(self.controls[j].vector()[:] - self.control_constraints[j][0]) < 0).nonzero()[0] for j in range(self.control_dim)]
		self.idx_active_upper = [(self.mu[j].vector()[:] + self.cc*(self.controls[j].vector()[:] - self.control_constraints[j][1]) > 0).nonzero()[0] for j in range(self.control_dim)]

		self.idx_active = [np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j])) for j in range(self.control_dim)]
		self.idx_active.sort()

		self.idx_inactive = [np.setdiff1d(np.arange(self.form_handler.control_spaces[j].dim()), self.idx_active[j] ) for j in range(self.control_dim)]

		return None



	def project_active(self, a, b):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[self.idx_active[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_active_lower(self, a, b):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active_lower[j]] = a[j].vector()[self.idx_active_lower[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_active_upper(self, a, b):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active_upper[j]] = a[j].vector()[self.idx_active_upper[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_inactive(self, a, b):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[self.idx_inactive[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def double_scalar_product(self, a1, a2, b1, b2):

		s1 = summ([fenics.assemble(fenics.inner(a1[j], b1[j])*self.form_handler.control_measures[j]) for j in range(self.control_dim)])
		s2 = summ([fenics.assemble(fenics.inner(a2[j], b2[j])*self.form_handler.control_measures[j]) for j in range(self.control_dim)])

		return s1 + s2




	def newton_solve(self):
		"""Solves the truncated Newton problem using an iterative method (cg, minres or cr)

		Returns
		-------
		self.delta_control : List[dolfin.function.function.Function]
			the Newton increment

		"""

		self.compute_sets()

		for j in range(self.control_dim):
			self.delta_control[j].vector()[:] = 0.0
			self.delta_mu[j].vector()[:] = 0.0
		self.gradient_problem.solve()

		self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ap1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ap2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

		### CG method
		if self.inner_newton == 'cg':

			for j in range(self.control_dim):
				self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0]
				self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1]

			self.project_active(self.mu, self.temp_storage)
			self.project_active_lower(self.temp_storage1, self.temp_storage1)
			self.project_active_upper(self.temp_storage2, self.temp_storage2)

			for j in range(self.control_dim):
				self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
				self.residual2[j].vector()[:] = - self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
				self.p1[j].vector()[:] = self.residual1[j].vector()[:]
				self.p2[j].vector()[:] = self.residual2[j].vector()[:]

			self.res_norm_squared = self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2)
			self.eps_0 = np.sqrt(self.res_norm_squared)

			for i in range(self.max_it_inner_newton):
				self.hessian_actions = self.hessian_application_simplified(self.p1)
				self.project_active(self.p2, self.temp_storage1)
				self.project_active(self.p1, self.temp_storage2)

				for j in range(self.control_dim):
					self.Ap1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage1[j].vector()[:]
					self.Ap2[j].vector()[:] = self.temp_storage2[j].vector()[:]


				self.eps = np.sqrt(self.res_norm_squared)
				# print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance:
					break

				### Problem: Matrix Not Positive Definitie!!! Need minres
				self.alpha = self.res_norm_squared / self.double_scalar_product(self.p1, self.p2, self.Ap1, self.Ap2)
				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p1[j].vector()[:]
					self.delta_mu[j].vector()[:] += self.alpha*self.p2[j].vector()[:]
					self.residual1[j].vector()[:] -= self.alpha*self.Ap1[j].vector()[:]
					self.residual2[j].vector()[:] -= self.alpha*self.Ap2[j].vector()[:]

				self.res_norm_squared = self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2)
				self.beta = self.res_norm_squared / pow(self.eps, 2)

				for j in range(self.control_dim):
					self.p1[j].vector()[:] = self.residual1[j].vector()[:] + self.beta*self.p1[j].vector()[:]
					self.p2[j].vector()[:] = self.residual2[j].vector()[:] + self.beta*self.p2[j].vector()[:]



		# elif self.inner_newton == 'minres':
		# 	for j in range(self.control_dim):
		# 		self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
		# 		self.p_prev[j].vector()[:] = self.residual[j].vector()[:]
		#
		# 	self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
		#
		# 	self.form_handler.project_inactive(self.p_prev, self.inactive_part)
		# 	self.hessian_actions = self.application_simplified(self.inactive_part)
		# 	self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
		# 	self.form_handler.project_active(self.p_prev, self.active_part)
		#
		# 	for j in range(self.control_dim):
		# 		self.s_prev[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]
		#
		# 	for i in range(self.max_it_inner_newton):
		# 		self.alpha = self.form_handler.scalar_product(self.residual, self.s_prev) / self.form_handler.scalar_product(self.s_prev, self.s_prev)
		#
		# 		for j in range(self.control_dim):
		# 			self.delta_control[j].vector()[:] += self.alpha*self.p_prev[j].vector()[:]
		# 			self.residual[j].vector()[:] -= self.alpha*self.s_prev[j].vector()[:]
		#
		# 		self.eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
		# 		# print('Eps (mr): ' + str(self.eps / self.eps_0) + ' (relative)')
		# 		if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
		# 			break
		#
		# 		for j in range(self.control_dim):
		# 			self.p[j].vector()[:] = self.s_prev[j].vector()[:]
		#
		# 		self.form_handler.project_inactive(self.s_prev, self.inactive_part)
		# 		self.hessian_actions = self.application_simplified(self.inactive_part)
		# 		self.form_handler.project_inactive(self.hessian_actions, self.inactive_part)
		# 		self.form_handler.project_active(self.s_prev, self.active_part)
		#
		# 		for j in range(self.control_dim):
		# 			self.s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]
		#
		# 		self.beta_prev = self.form_handler.scalar_product(self.s, self.s_prev) / self.form_handler.scalar_product(self.s_prev, self.s_prev)
		# 		if i==0:
		# 			self.beta_pprev = 0.0
		# 		else:
		# 			self.beta_pprev = self.form_handler.scalar_product(self.s, self.s_pprev) / self.form_handler.scalar_product(self.s_pprev, self.s_pprev)
		#
		# 		for j in range(self.control_dim):
		# 			self.p[j].vector()[:] -= self.beta_prev*self.p_prev[j].vector()[:] + self.beta_pprev*self.p_pprev[j].vector()[:]
		# 			self.s[j].vector()[:] -= self.beta_prev*self.s_prev[j].vector()[:] + self.beta_pprev*self.s_pprev[j].vector()[:]
		#
		# 			self.p_pprev[j].vector()[:] = self.p_prev[j].vector()[:]
		# 			self.s_pprev[j].vector()[:] = self.s_prev[j].vector()[:]
		#
		# 			self.p_prev[j].vector()[:] = self.p[j].vector()[:]
		# 			self.s_prev[j].vector()[:] = self.s[j].vector()[:]


		elif self.inner_newton == 'cr':
			self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ar1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ar2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ap1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ap2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

			for j in range(self.control_dim):
				self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0]
				self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1]

			self.project_active(self.mu, self.temp_storage)
			self.project_active_lower(self.temp_storage1, self.temp_storage1)
			self.project_active_upper(self.temp_storage2, self.temp_storage2)

			for j in range(self.control_dim):
				self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
				self.residual2[j].vector()[:] = -self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
				self.p1[j].vector()[:] = self.residual1[j].vector()[:]
				self.p2[j].vector()[:] = self.residual2[j].vector()[:]

			self.eps_0 = np.sqrt(self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))

			self.hessian_actions = self.hessian_application_simplified(self.residual1)
			self.project_active(self.residual1, self.temp_storage1)
			self.project_active(self.residual2, self.temp_storage2)

			for j in range(self.control_dim):
				self.Ar1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
				self.Ar2[j].vector()[:] = self.temp_storage1[j].vector()[:]
				self.Ap1[j].vector()[:] = self.Ar1[j].vector()[:]
				self.Ap2[j].vector()[:] = self.Ar2[j].vector()[:]

			self.rAr = self.double_scalar_product(self.residual1, self.residual2, self.Ar1, self.Ar2)

			for i in range(self.max_it_inner_newton):
				self.alpha = self.rAr / self.double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p1[j].vector()[:]
					self.delta_mu[j].vector()[:] += self.alpha*self.p2[j].vector()[:]
					self.residual1[j].vector()[:] -= self.alpha*self.Ap1[j].vector()[:]
					self.residual2[j].vector()[:] -= self.alpha*self.Ap2[j].vector()[:]

				self.eps = np.sqrt(self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))
				# print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
					break

				self.hessian_actions = self.hessian_application_simplified(self.residual1)
				self.project_active(self.residual1, self.temp_storage1)
				self.project_active(self.residual2, self.temp_storage2)

				for j in range(self.control_dim):
					self.Ar1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
					self.Ar2[j].vector()[:] = self.temp_storage1[j].vector()[:]

				self.rAr_new = self.double_scalar_product(self.residual1, self.residual2, self.Ar1, self.Ar2)
				self.beta = self.rAr_new / self.rAr

				for j in range(self.control_dim):
					self.p1[j].vector()[:] = self.residual1[j].vector()[:] + self.beta*self.p1[j].vector()[:]
					self.p2[j].vector()[:] = self.residual2[j].vector()[:] + self.beta*self.p2[j].vector()[:]
					self.Ap1[j].vector()[:] = self.Ar1[j].vector()[:] + self.beta*self.Ap1[j].vector()[:]
					self.Ap2[j].vector()[:] = self.Ar2[j].vector()[:] + self.beta*self.Ap2[j].vector()[:]

				self.rAr = self.rAr_new

		else:
			raise SystemExit('OptimizationRoutine.inner_newton needs to be one of cg, minres or cr.')

		return self.delta_control, self.delta_mu
