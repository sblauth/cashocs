"""
Created on 24/02/2020, 09.26

@author: blauths
"""

import fenics
from ..helpers import summ



class GradientProblem:
	
	def __init__(self, form_handler, state_problem, adjoint_problem):
		"""The class that handles the computation of the gradient
		
		Parameters
		----------
		form_handler : adpack.forms.FormHandler
			the FormHandler object of the optimization problem
		state_problem : adpack.pde_problems.state_problem.StateProblem
			the StateProblem object used to solve the state equations
		adjoint_problem : adpack.pde_problems.adjoint_problem.AdjointProblem
			the AdjointProblem used to solve the adjoint equations
		"""
		
		self.form_handler = form_handler
		self.state_problem = state_problem
		self.adjoint_problem = adjoint_problem
		
		self.gradients = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.config = self.form_handler.config
		
		self.has_solution = False
	
	
	def solve(self):
		"""Solves the Riesz projection problem in order to obtain the gradient of the cost functional
		
		Returns
		-------
		self.gradient : dolfin.function.function.Function
			the function representing the gradient of the (reduced) cost functional

		"""
		
		self.state_problem.solve()
		self.adjoint_problem.solve()
		
		if not self.has_solution:
			for i in range(self.form_handler.control_dim):
				A = fenics.assemble(self.form_handler.gradient_forms_lhs[i], keep_diagonal=True)
				A.ident_zeros()
				b = fenics.assemble(self.form_handler.gradient_forms_rhs[i])
				fenics.solve(A, self.gradients[i].vector(), b)
			
			self.has_solution = True
			
			self.gradient_norm_squared = summ([fenics.assemble(fenics.inner(self.gradients[i], self.gradients[i])*self.form_handler.control_measures[i]) for i in range(self.form_handler.control_dim)])
		
		return self.gradients
	
	
	
	def return_norm_squared(self):
		"""Returns the norm of the gradient squared, used e.g. in the Armijo line search
		
		Returns
		-------
		self.gradient_norm_squared : float
			|| `self.gradient` || in L^2 (corresponding to the domain given by control_measure)

		"""
		
		self.solve()
		
		return self.gradient_norm_squared
