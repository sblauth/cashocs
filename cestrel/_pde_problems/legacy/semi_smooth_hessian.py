"""
Created on 06/03/2020, 10.22

@author: blauths
"""

import fenics
import numpy as np
from .._exceptions import ConfigError



class SemiSmoothHessianProblem:
	"""A class that represents the problem for determining a search direction for a semi-smooth Newton method

	"""

	def __init__(self, form_handler, gradient_problem, control_constraints):
		"""The class that manages the computations for the Hessian in the truncated Newton method

		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			the FormHandler object for the optimization problem
		gradient_problem : cestrel._pde_problems.GradientProblem
			the GradientProblem object (we need the gradient for the computation of the Hessian)
		control_constraints : list[list[dolfin.function.function.Function]]
		"""

		self.form_handler = form_handler
		self.gradient_problem = gradient_problem
		self.control_constraints = control_constraints

		self.config = self.form_handler.config
		self.gradients = self.gradient_problem.gradients

		self.inner_newton = self.config.get('OptimizationRoutine', 'inner_newton', fallback='cr')
		self.max_it_inner_newton = self.config.getint('OptimizationRoutine', 'max_it_inner_newton', fallback=50)
		self.inner_newton_tolerance = self.config.getfloat('OptimizationRoutine', 'inner_newton_tolerance', fallback=1e-15)

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
			self.mu[j].vector()[:] = -1.0
		self.temp = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp_storage = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.cc = 1e-4

		self.inactive_part = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.active_part = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.controls = self.form_handler.controls

		self.no_sensitivity_solves = 0



	def __hessian_application(self):
		"""Returns the application of J''(u)[h] where h = self.test_direction

		This is needed in the truncated Newton method where we solve the system

		$$ J''(u) [\delta u] = - J'(u)
		$$

		via iterative methods (cg, minres, cr)

		Returns
		-------
		self.form_handler.hessian_action : List[dolfin.function.function.Function]
			the generic function that saves the result of J''(u)[h]

		"""

		self.states_prime = self.form_handler.states_prime
		self.adjoints_prime = self.form_handler.adjoints_prime
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		# TODO: Update this
		for i in range(self.state_dim):
			fenics.solve(self.form_handler.sensitivity_eqs_lhs[i]==self.form_handler.sensitivity_eqs_rhs[i], self.states_prime[i], self.bcs_list_ad[i])

		for i in range(self.state_dim):
			fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1-i]==self.form_handler.w_1[-1-i], self.adjoints_prime[-1-i], self.bcs_list_ad[-1-i])


		for i in range(self.control_dim):
			b = fenics.as_backend_type(fenics.assemble(self.form_handler.hessian_rhs[i])).vec()
			x = self.form_handler.hessian_actions[i].vector().vec()
			self.form_handler.ksps[i].solve(b, x)

			if self.form_handler.ksps[i].getConvergedReason() < 0:
				raise Exception('Krylov solver did not converge. Reason: ' + str(self.form_handler.ksps[i].getConvergedReason()))

		self.no_sensitivity_solves += 2

		return self.form_handler.hessian_actions



	def __hessian_application_simplified(self, x):
		"""A simplified version of the application of the Hessian.

		Computes J''(u)[x], where x is the input vector (see self.__hessian_application for more details)

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

		self.hessian_actions = self.__hessian_application()

		return self.hessian_actions



	def __double_scalar_product(self, a1, a2, b1, b2):
		"""Computes the scalar product for functions with two components.

		Parameters
		----------
		a1 : list[dolfin.function.function.Function]
			The first component of the first argument.
		a2 : list[dolfin.function.function.Function]
			The second component of the first argument.
		b1 : list[dolfin.function.function.Function]
			The first component of the second argument.
		b2 : list[dolfin.function.function.Function]
			The second component of the second argument.

		Returns
		-------
		float
			The value of the scalar product.
		"""

		s1 = self.form_handler.scalar_product(a1, b1)
		s2 = self.form_handler.scalar_product(a2, b2)

		return s1 + s2



	def newton_solve(self):
		"""Solves the truncated Newton problem using an iterative method (cg, minres or cr).

		Returns
		-------
		self.delta_control : List[dolfin.function.function.Function]
			The Newton increment.

		"""

		self.form_handler.compute_active_sets()

		for j in range(self.control_dim):
			self.delta_control[j].vector()[:] = 0.0
			self.delta_mu[j].vector()[:] = 0.0
		self.gradient_problem.solve()

		### CG method
		if self.inner_newton == 'cg':

			self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ap1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ap2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

			for j in range(self.control_dim):
				self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0].vector()[:]
				self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1].vector()[:]

			self.form_handler.restrict_to_active_set(self.mu, self.temp_storage)
			self.form_handler.restrict_to_lower_active_set(self.temp_storage1, self.temp_storage1)
			self.form_handler.restrict_to_upper_active_set(self.temp_storage2, self.temp_storage2)

			for j in range(self.control_dim):
				self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
				self.residual2[j].vector()[:] = - self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
				self.p1[j].vector()[:] = self.residual1[j].vector()[:]
				self.p2[j].vector()[:] = self.residual2[j].vector()[:]

			self.res_norm_squared = self.__double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2)
			self.eps_0 = np.sqrt(self.res_norm_squared)

			for i in range(self.max_it_inner_newton):
				self.hessian_actions = self.__hessian_application_simplified(self.p1)
				self.form_handler.restrict_to_active_set(self.p2, self.temp_storage1)
				self.form_handler.restrict_to_active_set(self.p1, self.temp_storage2)

				for j in range(self.control_dim):
					self.Ap1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage1[j].vector()[:]
					self.Ap2[j].vector()[:] = self.temp_storage2[j].vector()[:]


				self.eps = np.sqrt(self.res_norm_squared)
				print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance:
					break

				### Problem: Matrix Not Positive Definitie!!! Need minres
				self.alpha = self.res_norm_squared / self.__double_scalar_product(self.p1, self.p2, self.Ap1, self.Ap2)
				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p1[j].vector()[:]
					self.delta_mu[j].vector()[:] += self.alpha*self.p2[j].vector()[:]
					self.residual1[j].vector()[:] -= self.alpha*self.Ap1[j].vector()[:]
					self.residual2[j].vector()[:] -= self.alpha*self.Ap2[j].vector()[:]

				self.res_norm_squared = self.__double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2)
				self.beta = self.res_norm_squared / pow(self.eps, 2)

				for j in range(self.control_dim):
					self.p1[j].vector()[:] = self.residual1[j].vector()[:] + self.beta*self.p1[j].vector()[:]
					self.p2[j].vector()[:] = self.residual2[j].vector()[:] + self.beta*self.p2[j].vector()[:]



		elif self.inner_newton == 'minres':
			self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.p_prev1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.p_pprev1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.p_prev2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.p_pprev2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.s_prev1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.s_pprev1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.s_prev2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.s_pprev2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.s1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.s2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

			for j in range(self.control_dim):
				self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0].vector()[:]
				self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1].vector()[:]

			self.form_handler.restrict_to_active_set(self.mu, self.temp_storage)
			self.form_handler.restrict_to_lower_active_set(self.temp_storage1, self.temp_storage1)
			self.form_handler.restrict_to_upper_active_set(self.temp_storage2, self.temp_storage2)

			for j in range(self.control_dim):
				self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
				self.residual2[j].vector()[:] = -self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
				self.p_prev1[j].vector()[:] = self.residual1[j].vector()[:]
				self.p_prev2[j].vector()[:] = self.residual2[j].vector()[:]

			self.eps_0 = np.sqrt(self.__double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))

			self.hessian_actions = self.__hessian_application_simplified(self.p_prev1)
			self.form_handler.restrict_to_active_set(self.p_prev1, self.temp_storage1)
			self.form_handler.restrict_to_active_set(self.p_prev2, self.temp_storage2)

			for j in range(self.control_dim):
				self.s_prev1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
				self.s_prev2[j].vector()[:] = self.temp_storage1[j].vector()[:]

			for i in range(self.max_it_inner_newton):
				self.alpha = self.__double_scalar_product(self.residual1, self.residual2, self.s_prev1, self.s_prev2) / self.__double_scalar_product(self.s_prev1, self.s_prev2, self.s_prev1, self.s_prev2)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p_prev1[j].vector()[:]
					self.delta_mu[j].vector()[:] += self.alpha*self.p_prev2[j].vector()[:]

					self.residual1[j].vector()[:] -= self.alpha*self.s_prev1[j].vector()[:]
					self.residual2[j].vector()[:] -= self.alpha*self.s_prev2[j].vector()[:]

				self.eps = np.sqrt(self.__double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))
				# print('Eps (mr): ' + str(self.eps / self.eps_0) + ' (relative)')
				if self.eps/self.eps_0 < self.inner_newton_tolerance or i==self.max_it_inner_newton - 1:
					break

				for j in range(self.control_dim):
					self.p1[j].vector()[:] = self.s_prev1[j].vector()[:]
					self.p2[j].vector()[:] = self.s_prev2[j].vector()[:]

				self.hessian_actions = self.__hessian_application_simplified(self.s_prev1)
				self.form_handler.restrict_to_active_set(self.s_prev1, self.temp_storage1)
				self.form_handler.restrict_to_active_set(self.s_prev2, self.temp_storage2)

				for j in range(self.control_dim):
					self.s1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
					self.s2[j].vector()[:] = self.temp_storage1[j].vector()[:]

				self.beta_prev = self.__double_scalar_product(self.s1, self.s2, self.s_prev1, self.s_prev2) / self.__double_scalar_product(self.s_prev1, self.s_prev2, self.s_prev1, self.s_prev2)

				if i==0:
					self.beta_pprev = 0.0
				else:
					self.beta_pprev = self.__double_scalar_product(self.s1, self.s2, self.s_pprev1, self.s_pprev2) / self.__double_scalar_product(self.s_pprev1, self.s_pprev2, self.s_pprev1, self.s_pprev2)

				for j in range(self.control_dim):
					self.p1[j].vector()[:] -= self.beta_prev*self.s_prev1[j].vector()[:] + self.beta_pprev*self.p_prev1[j].vector()[:]
					self.p2[j].vector()[:] -= self.beta_prev*self.s_prev2[j].vector()[:] + self.beta_pprev*self.p_prev2[j].vector()[:]
					self.s1[j].vector()[:] -= self.beta_prev*self.s_prev1[j].vector()[:] + self.beta_pprev*self.s_pprev1[j].vector()[:]
					self.s2[j].vector()[:] -= self.beta_prev*self.s_prev2[j].vector()[:] + self.beta_pprev*self.s_pprev2[j].vector()[:]

					self.p_pprev1[j].vector()[:] = self.p_prev1[j].vector()[:]
					self.p_pprev2[j].vector()[:] = self.p_prev2[j].vector()[:]
					self.s_pprev1[j].vector()[:] = self.s_prev1[j].vector()[:]
					self.s_pprev2[j].vector()[:] = self.s_prev2[j].vector()[:]

					self.p_prev1[j].vector()[:] = self.p1[j].vector()[:]
					self.p_prev2[j].vector()[:] = self.p2[j].vector()[:]
					self.s_prev1[j].vector()[:] = self.s1[j].vector()[:]
					self.s_prev2[j].vector()[:] = self.s2[j].vector()[:]


		elif self.inner_newton == 'cr':
			self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ar1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ar2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ap1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
			self.Ap2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

			for j in range(self.control_dim):
				self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0].vector()[:]
				self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1].vector()[:]

			self.form_handler.restrict_to_active_set(self.mu, self.temp_storage)
			self.form_handler.restrict_to_lower_active_set(self.temp_storage1, self.temp_storage1)
			self.form_handler.restrict_to_upper_active_set(self.temp_storage2, self.temp_storage2)

			for j in range(self.control_dim):
				self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
				self.residual2[j].vector()[:] = -self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
				self.p1[j].vector()[:] = self.residual1[j].vector()[:]
				self.p2[j].vector()[:] = self.residual2[j].vector()[:]

			self.eps_0 = np.sqrt(self.__double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))

			self.hessian_actions = self.__hessian_application_simplified(self.residual1)
			self.form_handler.restrict_to_active_set(self.residual1, self.temp_storage1)
			self.form_handler.restrict_to_active_set(self.residual2, self.temp_storage2)

			for j in range(self.control_dim):
				self.Ar1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
				self.Ar2[j].vector()[:] = self.temp_storage1[j].vector()[:]
				self.Ap1[j].vector()[:] = self.Ar1[j].vector()[:]
				self.Ap2[j].vector()[:] = self.Ar2[j].vector()[:]

			self.rAr = self.__double_scalar_product(self.residual1, self.residual2, self.Ar1, self.Ar2)

			for i in range(self.max_it_inner_newton):
				# self.alpha = self.__double_scalar_product(self.residual1, self.residual2, self.Ap1, self.Ap2) / self.__double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)
				self.alpha = self.rAr / self.__double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.alpha*self.p1[j].vector()[:]
					self.delta_mu[j].vector()[:] += self.alpha*self.p2[j].vector()[:]

					self.residual1[j].vector()[:] -= self.alpha*self.Ap1[j].vector()[:]
					self.residual2[j].vector()[:] -= self.alpha*self.Ap2[j].vector()[:]

				self.eps = np.sqrt(self.__double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))
				print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
				if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
					break

				self.hessian_actions = self.__hessian_application_simplified(self.residual1)
				self.form_handler.restrict_to_active_set(self.residual1, self.temp_storage1)
				self.form_handler.restrict_to_active_set(self.residual2, self.temp_storage2)

				for j in range(self.control_dim):
					self.Ar1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
					self.Ar2[j].vector()[:] = self.temp_storage1[j].vector()[:]

				self.rAr_new = self.__double_scalar_product(self.residual1, self.residual2, self.Ar1, self.Ar2)
				self.beta = self.rAr_new / self.rAr
				# self.beta = -self.__double_scalar_product(self.Ar1, self.Ar2, self.Ap1, self.Ap2) / self.__double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)

				for j in range(self.control_dim):
					self.p1[j].vector()[:] = self.residual1[j].vector()[:] + self.beta*self.p1[j].vector()[:]
					self.p2[j].vector()[:] = self.residual2[j].vector()[:] + self.beta*self.p2[j].vector()[:]

					self.Ap1[j].vector()[:] = self.Ar1[j].vector()[:] + self.beta*self.Ap1[j].vector()[:]
					self.Ap2[j].vector()[:] = self.Ar2[j].vector()[:] + self.beta*self.Ap2[j].vector()[:]

				self.rAr = self.rAr_new

		else:
			raise ConfigError('Not a valid choice for OptimizationRoutine.inner_newton. Needs to be one of cg, minres or cr.')

		return self.delta_control, self.delta_mu
