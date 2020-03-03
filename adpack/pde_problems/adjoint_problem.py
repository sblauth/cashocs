"""
Created on 24/02/2020, 09.24

@author: blauths
"""

import fenics
import numpy as np



class AdjointProblem:
	
	def __init__(self, form_handler, state_problem):
		"""A class that implements the adjoint system, used e.g. to determine the gradient of the cost functional
		
		Parameters
		----------
		form_handler : adpack.forms.FormHandler
			the FormHandler object for the optimization problem
		state_problem : adpack.pde_problems.state_problem.StateProblem
			the StateProblem object used to get the point where we linearize the problem
		"""
		
		self.form_handler = form_handler
		self.state_problem = state_problem
		
		self.config = self.form_handler.config
		self.adjoints = self.form_handler.adjoints
		self.bcs_list_ad = self.form_handler.bcs_list_ad
		
		self.number_of_solves = 0
		self.has_solution = False
	
	
	
	def solve(self):
		"""Solves the adjoint system
		
		Returns
		-------
		self.adjoint : dolfin.function.function.Function
			the Function representing the solution of the adjoint system

		"""
		
		self.state_problem.solve()

		if not self.has_solution:
			for i in range(self.form_handler.state_dim):
				a, L = fenics.system(self.form_handler.adjoint_eq_forms[-1-i])
				fenics.solve(a==L, self.adjoints[-1-i], self.bcs_list_ad[-1-i])
		
			self.has_solution = True
			self.number_of_solves += 1
		
		return self.adjoints
