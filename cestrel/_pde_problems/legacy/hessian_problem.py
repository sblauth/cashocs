"""
Created on 24/02/2020, 16.48

@author: blauths
"""

import fenics
import numpy as np
from .._exceptions import NotConvergedError, ConfigError



class HessianProblem:
	"""Computes the action of the Hessian on some direction for truncated Newton methods.

	"""

	def __init__(self, form_handler, gradient_problem):
		"""Initializes the HessianProblem.
		
		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			The FormHandler object for the optimization problem.
		gradient_problem : cestrel._pde_problems.GradientProblem
			The GradientProblem object (this is needed for the computation
			of the Hessian).
		"""
		
		self.form_handler = form_handler
		self.gradient_problem = gradient_problem

		self.config = self.form_handler.config
		self.gradients = self.gradient_problem.gradients
		
		self.inner_newton = self.config.get('OptimizationRoutine', 'inner_newton', fallback='cr')
		self.max_it_inner_newton = self.config.getint('OptimizationRoutine', 'max_it_inner_newton', fallback=50)
		# TODO: Add absolute tolerance, too
		self.inner_newton_tolerance = self.config.getfloat('OptimizationRoutine', 'inner_newton_tolerance', fallback=1e-15)
		
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
		self.atol = self.config.getfloat('StateEquation', 'picard_atol', fallback=1e-20)
		self.maxiter = self.config.getint('StateEquation', 'picard_iter', fallback=50)
		self.picard_verbose = self.config.getboolean('StateEquation', 'picard_verbose', fallback=False)
		
		self.no_sensitivity_solves = 0

	
	
	def __hessian_application(self, x):
		r"""Computes the application of the Hessian to some element

		This is needed in the truncated Newton method where we solve the system

		$$ J''(u) [\delta u] = - J'(u)
		$$

		via iterative methods (cg, minres, cr)

		Parameters
		----------
		x : list[dolfin.function.function.Function]
			A function to which we want to apply the Hessian to.

		Returns
		-------
		list[dolfin.function.function.Function]
			The generic function that saves the result of the
			 application J''(u)[h].
		"""

		for i in range(self.control_dim):
			self.test_directions[i].vector()[:] = x[i].vector()[:]

		self.states_prime = self.form_handler.states_prime
		self.adjoints_prime = self.form_handler.adjoints_prime
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		if not self.config.getboolean('StateEquation', 'picard_iteration', fallback=False):

			for i in range(self.state_dim):
				# TODO: Update this
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
					raise NotConvergedError('Failed to solve the Picard iteration.')

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
					raise NotConvergedError('Failed to solve the Picard iteration.')

				for j in range(self.form_handler.state_dim):
					fenics.solve(self.form_handler.adjoint_sensitivity_eqs_lhs[-1-j]==self.form_handler.w_1[-1-j], self.adjoints_prime[-1 - j], self.bcs_list_ad[-1 - j])


			if self.picard_verbose:
				print('')


		for i in range(self.control_dim):
			# TODO: Update this
			b = fenics.as_backend_type(fenics.assemble(self.form_handler.hessian_rhs[i])).vec()
			x = self.form_handler.hessian_actions[i].vector().vec()
			self.form_handler.ksps[i].solve(b, x)

			if self.form_handler.ksps[i].getConvergedReason() < 0:
				raise Exception('Krylov solver did not converge. Reason: ' + str(self.form_handler.ksps[i].getConvergedReason()))

		self.no_sensitivity_solves += 2
		
		return self.form_handler.hessian_actions

	
	
	def newton_solve(self):
		"""Solves the truncated Newton problem using an iterative method (cg, minres or cr).
		
		Returns
		-------
		delta_control : list[dolfin.function.function.Function]
			The Newton increment.
		"""

		self.form_handler.compute_active_sets()
		
		for j in range(self.control_dim):
			self.delta_control[j].vector()[:] = 0.0
		self.gradient_problem.solve()
		
		# CG method
		if self.inner_newton == 'cg':
			
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
				self.__p[j].vector()[:] = self.residual[j].vector()[:]
			
			self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
			self.__eps_0 = np.sqrt(self.res_norm_squared)
			
			for i in range(self.max_it_inner_newton):
				self.form_handler.restrict_to_inactive_set(self.__p, self.inactive_part)
				self.hessian_actions = self.__hessian_application(self.inactive_part)
				self.form_handler.restrict_to_inactive_set(self.hessian_actions, self.inactive_part)
				self.form_handler.restrict_to_active_set(self.__p, self.active_part)

				for j in range(self.control_dim):
					self.__q[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]
				
				self.__eps = np.sqrt(self.res_norm_squared)
				# print('Eps (CG): ' + str(self.__eps / self.__eps_0) + ' (rel)')
				if self.__eps / self.__eps_0 < self.inner_newton_tolerance:
					break
				
				self.__alpha = self.res_norm_squared / self.form_handler.scalar_product(self.__p, self.__q)
				for j in range(self.control_dim):
					self.delta_control[j].vector()[:] += self.__alpha * self.__p[j].vector()[:]
					self.residual[j].vector()[:] -= self.__alpha * self.__q[j].vector()[:]
				
				self.res_norm_squared = self.form_handler.scalar_product(self.residual, self.residual)
				self.__beta = self.res_norm_squared / pow(self.__eps, 2)
				
				for j in range(self.control_dim):
					self.__p[j].vector()[:] = self.residual[j].vector()[:] + self.__beta * self.__p[j].vector()[:]

		# minres method
		elif self.inner_newton == 'minres':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
				self.__p_prev[j].vector()[:] = self.residual[j].vector()[:]

			self.__eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.form_handler.restrict_to_inactive_set(self.__p_prev, self.inactive_part)
			self.hessian_actions = self.__hessian_application(self.inactive_part)
			self.form_handler.restrict_to_inactive_set(self.hessian_actions, self.inactive_part)
			self.form_handler.restrict_to_active_set(self.__p_prev, self.active_part)

			for j in range(self.control_dim):
				self.__s_prev[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]

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

				self.form_handler.restrict_to_inactive_set(self.__s_prev, self.inactive_part)
				self.hessian_actions = self.__hessian_application(self.inactive_part)
				self.form_handler.restrict_to_inactive_set(self.hessian_actions, self.inactive_part)
				self.form_handler.restrict_to_active_set(self.__s_prev, self.active_part)

				for j in range(self.control_dim):
					self.__s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]

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

		# cr method
		elif self.inner_newton == 'cr':
			for j in range(self.control_dim):
				self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
				self.__p[j].vector()[:] = self.residual[j].vector()[:]

			self.__eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

			self.form_handler.restrict_to_inactive_set(self.residual, self.inactive_part)
			self.hessian_actions = self.__hessian_application(self.inactive_part)
			self.form_handler.restrict_to_inactive_set(self.hessian_actions, self.inactive_part)
			self.form_handler.restrict_to_active_set(self.residual, self.active_part)

			for j in range(self.control_dim):
				self.__s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]
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
				if self.__eps / self.__eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
					break

				self.form_handler.restrict_to_inactive_set(self.residual, self.inactive_part)
				self.hessian_actions = self.__hessian_application(self.inactive_part)
				self.form_handler.restrict_to_inactive_set(self.hessian_actions, self.inactive_part)
				self.form_handler.restrict_to_active_set(self.residual, self.active_part)
				for j in range(self.control_dim):
					self.__s[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]

				self.__rAr_new = self.form_handler.scalar_product(self.residual, self.__s)
				self.__beta = self.__rAr_new / self.__rAr

				for j in range(self.control_dim):
					# self.__q_prev[j].vector()[:] = self.__q[j].vector()[:]
					self.__p[j].vector()[:] = self.residual[j].vector()[:] + self.__beta * self.__p[j].vector()[:]
					self.__q[j].vector()[:] = self.__s[j].vector()[:] + self.__beta * self.__q[j].vector()[:]


				self.__rAr = self.__rAr_new

		else:
			raise ConfigError('Not a valid choice for OptimizationRoutine.inner_newton. Needs to be one of cg, minres or cr.')
		
		return self.delta_control
