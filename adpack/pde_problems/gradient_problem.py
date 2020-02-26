"""
Created on 24/02/2020, 09.26

@author: blauths
"""

import fenics
import numpy as np



class GradientProblem:
	def __init__(self, form_handler, state_problem, adjoint_problem):
		self.form_handler = form_handler
		self.state_problem = state_problem
		self.adjoint_problem = adjoint_problem
		
		self.gradient = fenics.Function(self.form_handler.control_space)
		self.config = self.form_handler.config
		
		self.has_solution = False
	
	
	def solve(self):
		self.state_problem.solve()
		self.adjoint_problem.solve()
		
		if not self.has_solution:
			A = fenics.assemble(self.form_handler.gradient_form_lhs, keep_diagonal=True)
			A.ident_zeros()
			b = fenics.assemble(self.form_handler.gradient_form_rhs)
			fenics.solve(A, self.gradient.vector(), b)
			
			self.has_solution = True
			
			self.gradient_norm_squared = fenics.assemble(fenics.inner(self.gradient, self.gradient)*self.form_handler.control_measure)
		
		return self.gradient
	
	
	
	def return_norm_squared(self):
		self.solve()
		
		return self.gradient_norm_squared
