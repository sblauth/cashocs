"""
Created on 24/02/2020, 16.48

@author: blauths
"""

import fenics
import numpy as np


class HessianProblem:
	def __init__(self, form_handler, gradient_problem):
		self.form_handler = form_handler
		self.gradient_problem = gradient_problem
		
		self.config = self.form_handler.config
		self.gradient = self.gradient_problem.gradient
		
		self.inner_newton = self.config.get('OptimizationRoutine', 'inner_newton')
		
		self.test_direction = self.form_handler.test_direction
		self.res = fenics.Function(self.test_direction.function_space())
		self.p_prev = fenics.Function(self.test_direction.function_space())
		self.delta_control = fenics.Function(self.test_direction.function_space())
		
	
	
	def hessian_application(self):
		self.state_prime = self.form_handler.state_prime
		self.adjoint_prime = self.form_handler.adjoint_prime
		self.bcs_ad = self.form_handler.bcs_ad
		
		fenics.solve(self.form_handler.sensitivity_eq_lhs==self.form_handler.sensitivity_eq_rhs, self.state_prime, self.bcs_ad)
		
		fenics.solve(self.form_handler.adjoint_sensitivity_lhs==self.form_handler.w_1, self.adjoint_prime, self.bcs_ad)
		
		A = fenics.assemble(self.form_handler.hessian_lhs, keep_diagonal=True)
		A.ident_zeros()
		b = fenics.assemble(self.form_handler.hessian_rhs)
		fenics.solve(A, self.form_handler.hessian_action.vector(), b)
		
		return self.form_handler.hessian_action
	
	
	
	def application_simplified(self, x):
		self.test_direction.vector()[:] = x.vector()[:]
		self.hessian_action = self.hessian_application()
		
		return self.hessian_action
	
	
	
	def newton_solve(self):
		self.delta_control.vector()[:] = 0.0
		self.gradient_problem.solve()
		
		if self.inner_newton == 'cg':
			self.res.vector()[:] = self.gradient.vector()[:]
			self.res_norm_squared = fenics.assemble(fenics.inner(self.res, self.res)*self.form_handler.control_measure)
			self.eps_0 = np.sqrt(self.res_norm_squared)
			self.test_direction.vector()[:] = -self.res.vector()[:]
			
			for i in range(15):
				self.hessian_action = self.hessian_application()
				
				self.eps = np.sqrt(self.res_norm_squared)
				# print('Eps (CG): ' + str(self.eps))
				if self.eps / self.eps_0 < 1e-5:
					break
				
				self.alpha = self.res_norm_squared / fenics.assemble(fenics.inner(self.test_direction, self.hessian_action)*self.form_handler.control_measure)
				self.delta_control.vector()[:] += self.alpha*self.test_direction.vector()[:]
				self.res.vector()[:] += self.alpha*self.hessian_action.vector()[:]
				
				self.res_norm_squared = fenics.assemble(fenics.inner(self.res, self.res)*self.form_handler.control_measure)
				self.beta = self.res_norm_squared / pow(self.eps, 2)
				
				self.test_direction.vector()[:] = -self.res.vector()[:] + self.beta*self.test_direction.vector()[:]
		
		
		elif self.inner_newton == 'minres':
			self.delta_control.vector()[:] = 0.0

			self.p = fenics.Function(self.test_direction.function_space())
			self.p_prev = fenics.Function(self.test_direction.function_space())
			self.p_pprev = fenics.Function(self.test_direction.function_space())
			self.s_prev = fenics.Function(self.test_direction.function_space())
			self.s_pprev = fenics.Function(self.test_direction.function_space())

			self.res.vector()[:] = - self.gradient.vector()[:]
			self.eps_0 = np.sqrt(fenics.assemble(fenics.inner(self.res, self.res)*self.form_handler.control_measure))
			self.p_prev.vector()[:] = self.res.vector()[:]
			self.test_direction.vector()[:] = self.p_prev.vector()[:]
			self.hessian_action = self.hessian_application()
			self.s_prev.vector()[:] = self.hessian_action.vector()[:]

			for i in range(15):
				self.alpha = fenics.assemble(fenics.inner(self.res, self.s_prev)*self.form_handler.control_measure) / \
							 fenics.assemble(fenics.inner(self.s_prev, self.s_prev)*self.form_handler.control_measure)

				self.delta_control.vector()[:] += self.alpha*self.p_prev.vector()[:]
				self.res.vector()[:] -= self.alpha*self.s_prev.vector()[:]

				self.eps = np.sqrt(fenics.assemble(fenics.inner(self.res, self.res)*self.form_handler.control_measure))
				# print('Eps (minres): ' + str(self.eps))
				if self.eps / self.eps_0 < 1e-5:
					break

				self.p.vector()[:] = self.s_prev.vector()[:]

				self.test_direction.vector()[:] = self.s_prev.vector()[:]
				self.hessian_action = self.hessian_application()

				self.beta = fenics.assemble(fenics.inner(self.hessian_action, self.s_prev)*self.form_handler.control_measure) / \
							fenics.assemble(fenics.inner(self.s_prev, self.s_prev)*self.form_handler.control_measure)

				if i > 0:
					self.beta_prev = fenics.assemble(fenics.inner(self.hessian_action, self.s_pprev)*self.form_handler.control_measure) / \
								fenics.assemble(fenics.inner(self.s_pprev, self.s_pprev)*self.form_handler.control_measure)
				else:
					self.beta_prev = 0.0

				self.p.vector()[:] -= self.beta*self.p_prev.vector()[:] + self.beta_prev*self.p_pprev.vector()[:]
				self.hessian_action.vector()[:] -= self.beta*self.s_prev.vector()[:] + self.beta_prev*self.s_pprev.vector()[:]

				self.p_pprev.vector()[:] = self.p_prev.vector()[:]
				self.p_prev.vector()[:] = self.p.vector()[:]

				self.s_pprev.vector()[:] = self.s_prev.vector()[:]
				self.s_prev.vector()[:] = self.hessian_action.vector()[:]

				self.beta_prev = self.beta
		
		elif self.inner_newton == 'cr':
			self.res.vector()[:] = - self.gradient.vector()[:]
			self.eps_0 = np.sqrt(fenics.assemble(fenics.inner(self.res, self.res)*self.form_handler.control_measure))

			self.beta = 0

			self.p = fenics.Function(self.test_direction.function_space())
			self.p_prev = fenics.Function(self.test_direction.function_space())

			self.q = fenics.Function(self.test_direction.function_space())
			self.q_prev = fenics.Function(self.test_direction.function_space())
			
			self.s = fenics.Function(self.test_direction.function_space())
			self.s_new = fenics.Function(self.test_direction.function_space())
			
			self.s.vector()[:] = self.application_simplified(self.res).vector()[:]

			self.res_new = fenics.Function(self.test_direction.function_space())

			for i in range(15):
				self.p.vector()[:] = self.res.vector()[:] + self.beta*self.p_prev.vector()[:]
				
				self.q.vector()[:] = self.s.vector()[:] + self.beta*self.q_prev.vector()[:]
				
				self.alpha = fenics.assemble(fenics.inner(self.res, self.s)*self.form_handler.control_measure) / \
							fenics.assemble(fenics.inner(self.q, self.q)*self.form_handler.control_measure)
				
				self.delta_control.vector()[:] += self.alpha*self.p.vector()[:]
				self.res_new.vector()[:] = self.res.vector()[:] - self.alpha*self.q.vector()[:]

				self.eps = np.sqrt(fenics.assemble(fenics.inner(self.res_new, self.res_new)*self.form_handler.control_measure))
				# print('Eps (cr): ' + str(self.eps))

				if self.eps / self.eps_0 < 1e-5:
					break

				self.s_new.vector()[:] = self.application_simplified(self.res_new).vector()[:]

				self.beta = fenics.assemble(fenics.inner(self.res_new, self.s_new)*self.form_handler.control_measure) / \
							fenics.assemble(fenics.inner(self.res, self.q)*self.form_handler.control_measure)
				
				self.s.vector()[:] = self.s_new.vector()[:]
				self.p_prev.vector()[:] = self.p.vector()[:]
				self.q_prev.vector()[:] = self.q.vector()[:]
				self.res.vector()[:] = self.res_new.vector()[:]

		else:
			raise SystemExit('OptimizationRoutine.inner_newton needs to be one of cg, minres or cr.')
		
		return self.delta_control
