"""
Created on 01.04.20, 15:15

@author: sebastian
"""

import fenics
import numpy as np
from .._exceptions import NotConvergedError, ConfigError



class UnconstrainedHessianProblem:
	"""This class solves an unconstrained Hessian problem for a truncated Newton method.

	This is the unconstrained version used when a Newton method is chosen as
	inner solver for the Primal-Dual-Active-Set method.
	"""

	def __init__(self, form_handler, gradient_problem):
		"""Initializes the object.

		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			The FormHandler object for the optimization problem.
		gradient_problem : cestrel._pde_problems.GradientProblem
			The corresponding GradientProblem object.
		"""

		self.form_handler = form_handler
		self.gradient_problem = gradient_problem

		self.config = self.form_handler.config
		self.gradients = self.gradient_problem.gradients
		self.reduced_gradient = [fenics.Function(self.form_handler.control_spaces[j]) for j in range(len(self.gradients))]

		self.inner_newton = self.config.get('OptimizationRoutine', 'inner_newton', fallback='cr')
		self.max_it_inner_newton = self.config.getint('OptimizationRoutine', 'max_it_inner_newton', fallback=50)
		self.inner_newton_tolerance = self.config.getfloat('OptimizationRoutine', 'inner_newton_tolerance', fallback=1e-10)

		self.test_directions = self.form_handler.test_directions
		self.residual = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.delta_control = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.state_dim = self.form_handler.state_dim
		self.control_dim = self.form_handler.control_dim

		self.__p = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__p_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__p_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__s = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__s_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__s_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__q = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__q_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.__r_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.inactive_part = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.active_part = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.controls = self.form_handler.controls

		self.rtol = self.config.getfloat('StateEquation', 'picard_rtol', fallback=1e-10)
		self.atol = self.config.getfloat('StateEquation', 'picard_atol', fallback=1e-12)
		self.maxiter = self.config.getint('StateEquation', 'picard_iter', fallback=50)
		self.picard_verbose = self.config.getboolean('StateEquation', 'picard_verbose', fallback=False)

		self.no_sensitivity_solves = 0



	def __hessian_application(self):
		"""Returns the application of the Hessian to some element.

		Returns the application of J''(u)[h] where h = self.test_direction.
		This is needed in the truncated Newton method where we solve the system

			J''(u) du = - J'(u)

		via iterative methods (cg, minres, cr).

		Returns
		-------
		list[dolfin.function.function.Function]
			The generic function that saves the result of J''(u)[h].
		"""

		self.states_prime = self.form_handler.states_prime
		self.adjoints_prime = self.form_handler.adjoints_prime
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		if not self.form_handler.state_is_picard:

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
					raise NotConvergedError('Failed to solve the Picard Iteration')

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
					raise NotConvergedError('Failed to solve the Picard Iteration')

				for j in range(self.form_handler.state_dim):
					fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1 - j]==self.form_handler.w_1[-1 - j], self.adjoints_prime[-1 - j], self.bcs_list_ad[-1 - j])

			if self.picard_verbose:
				print('')

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

		Computes J''(u)[x], where x is the input vector (see self.__hessian_application for more details).

		Parameters
		----------
		x : list[dolfin.function.function.Function]
			A function to which we want to apply the Hessian to.

		Returns
		-------
		list[dolfin.function.function.Function]
			The generic function that saves the result of J''(u)[h].

		"""

		for i in range(self.control_dim):
			self.test_directions[i].vector()[:] = x[i].vector()[:]

		self.hessian_actions = self.__hessian_application()

		return self.hessian_actions



	def newton_solve(self, idx_active):
		"""Solves the truncated Newton problem using an iterative method (cg, minres or cr).

		Returns
		-------
		self.delta_control : list[dolfin.function.function.Function]
			The Newton increment.
		"""

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
				self.__p[j].vector()[:] = self.residual[j].vector()[:]

			self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
			self.__eps_0 = np.sqrt(self.res_norm_squared)

			for i in range(self.max_it_inner_newton):

				for j in range(self.control_dim):
					self.__p[j].vector()[idx_active[j]] = 0.0

				self.hessian_actions = self.__hessian_application_simplified(self.__p)

				for j in range(self.control_dim):
					self.__q[j].vector()[:] = self.hessian_actions[j].vector()[:]
					self.__q[j].vector()[idx_active[j]] = 0.0

				self.__eps = np.sqrt(self.res_norm_squared)
				# print('Eps (CG): ' + str(self.__eps / self.__eps_0) + ' (rel)')
				if self.__eps/self.__eps_0 < self.inner_newton_tolerance:
					break

				self.__alpha = self.res_norm_squared / self.form_handler.scalar_product(self.__p, self.__q)
				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.__alpha * self.__p[j].vector()[:]
					self.residual[j].vector()[:] -= self.__alpha * self.__q[j].vector()[:]

				self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
				self.__beta = self.res_norm_squared / pow(self.__eps, 2)

				for j in range(self.control_dim):
					self.__p[j].vector()[:] = self.residual[j].vector()[:] + self.__beta * self.__p[j].vector()[:]


		elif self.inner_newton == 'minres':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
				self.__p_prev[j].vector()[:] = self.residual[j].vector()[:]

			self.__eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.hessian_actions = self.__hessian_application_simplified(self.__p_prev)

			for j in range(self.control_dim):
				self.__s_prev[j].vector()[:] = self.hessian_actions[j].vector()[:]
				self.__s_prev[j].vector()[idx_active[j]] = 0.0

			for i in range(self.max_it_inner_newton):
				self.__alpha = self.form_handler.scalar_product(self.residual, self.__s_prev) / self.form_handler.scalar_product(self.__s_prev, self.__s_prev)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.__alpha * self.__p_prev[j].vector()[:]
					self.residual[j].vector()[:] -= self.__alpha * self.__s_prev[j].vector()[:]

				self.__eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
				# print('Eps (mr): ' + str(self.__eps / self.__eps_0) + ' (relative)')
				if self.__eps / self.__eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
					break

				for j in range(self.control_dim):
					self.__p[j].vector()[:] = self.__s_prev[j].vector()[:]

				self.hessian_actions = self.__hessian_application_simplified(self.__s_prev)

				for j in range(self.control_dim):
					self.__s[j].vector()[:] = self.hessian_actions[j].vector()[:]
					self.__s[j].vector()[idx_active[j]] = 0.0

				self.__beta_prev = self.form_handler.scalar_product(self.__s, self.__s_prev) / self.form_handler.scalar_product(self.__s_prev, self.__s_prev)
				if i==0:
					self.__beta_pprev = 0.0
				else:
					self.__beta_pprev = self.form_handler.scalar_product(self.__s, self.__s_pprev) / self.form_handler.scalar_product(self.__s_pprev, self.__s_pprev)

				for j in range(self.control_dim):
					self.__p[j].vector()[:] -= self.__beta_prev * self.__p_prev[j].vector()[:] + self.__beta_pprev * self.__p_pprev[j].vector()[:]
					self.__s[j].vector()[:] -= self.__beta_prev * self.__s_prev[j].vector()[:] + self.__beta_pprev * self.__s_pprev[j].vector()[:]

					self.__p_pprev[j].vector()[:] = self.__p_prev[j].vector()[:]
					self.__s_pprev[j].vector()[:] = self.__s_prev[j].vector()[:]

					self.__p_prev[j].vector()[:] = self.__p[j].vector()[:]
					self.__s_prev[j].vector()[:] = self.__s[j].vector()[:]


		elif self.inner_newton=='cr':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
				self.__p[j].vector()[:] = self.residual[j].vector()[:]

			self.__eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.hessian_actions = self.__hessian_application_simplified(self.residual)

			for j in range(self.control_dim):
				self.__s[j].vector()[:] = self.hessian_actions[j].vector()[:]
				self.__s[j].vector()[idx_active[j]] = 0.0
				self.__q[j].vector()[:] = self.__s[j].vector()[:]

			self.__rAr = self.form_handler.scalar_product(self.residual, self.__s)

			for i in range(self.max_it_inner_newton):
				self.__alpha = self.__rAr / self.form_handler.scalar_product(self.__q, self.__q)

				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.__alpha * self.__p[j].vector()[:]
					self.__r_prev[j].vector()[:] = self.residual[j].vector()[:]
					self.residual[j].vector()[:] -= self.__alpha * self.__q[j].vector()[:]

				self.__eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
				# print('Eps (cr): ' + str(self.__eps / self.__eps_0) + ' (relative)')
				if self.__eps/self.__eps_0 < self.inner_newton_tolerance or i==self.max_it_inner_newton - 1:
					break

				self.hessian_actions = self.__hessian_application_simplified(self.residual)

				for j in range(self.control_dim):
					self.__s[j].vector()[:] = self.hessian_actions[j].vector()[:]
					self.__s[j].vector()[idx_active[j]] = 0.0

				self.__rAr_new = self.form_handler.scalar_product(self.residual, self.__s)
				self.__beta = self.__rAr_new / self.__rAr

				for j in range(self.control_dim):
					self.__q_prev[j].vector()[:] = self.__q[j].vector()[:]
					self.__q[j].vector()[:] = self.__s[j].vector()[:] + self.__beta * self.__q[j].vector()[:]
					self.__p[j].vector()[:] = self.residual[j].vector()[:] + self.__beta * self.__p[j].vector()[:]

				self.__rAr = self.__rAr_new

		else:
			raise ConfigError('Not a valid choice for OptimizationRoutine.inner_newton. Needs to be one of cg, minres or cr.')

		return self.delta_control
