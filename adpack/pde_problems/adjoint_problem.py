"""
Created on 24/02/2020, 09.24

@author: blauths
"""

import fenics
import numpy as np



class AdjointProblem:
	def __init__(self, form_handler, state_problem):
		self.form_handler = form_handler
		self.state_problem = state_problem
		
		self.config = self.form_handler.config
		self.adjoint = self.form_handler.adjoint
		self.bcs_ad = self.form_handler.bcs_ad
		
		self.number_of_solves = 0
		self.has_solution = False
	
	
	
	def solve(self):
		self.state_problem.solve()

		if not self.has_solution:
			a, L = fenics.system(self.form_handler.adjoint_eq_form)
			fenics.solve(a==L, self.adjoint, self.bcs_ad)
		
			self.has_solution = True
			self.number_of_solves += 1
		
		return self.adjoint
