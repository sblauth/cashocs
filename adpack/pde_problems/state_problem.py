"""
Created on 24/02/2020, 09.19

@author: blauths
"""

import fenics
import numpy as np



class StateProblem:
	
	def __init__(self, form_handler):
		"""An abstract representation of the state system, which is also used to solve it
		
		Parameters
		----------
		form_handler : adpack.forms.FormHandler
			the FormHandler of the optimization problem
		"""
		
		self.form_handler = form_handler
		
		self.config = self.form_handler.config
		self.bcs = self.form_handler.bcs
		self.state = self.form_handler.state
		self.control = self.form_handler.control
		
		self.number_of_solves = 0
		self.has_solution = False
	
	
	
	def solve(self):
		"""Solve command for the state equation. Solution is "stored" in self.state (corresponding to user input)
		
		Returns
		-------
		self.state : dolfin.function.function.Function
			the function that corresponds to the solution of the state system

		"""
		
		if not self.has_solution:
			if self.config.getboolean('StateEquation', 'is_linear'):
				a, L = fenics.system(self.form_handler.state_eq_form)
				fenics.solve(a==L, self.state, self.bcs)
				
			else:
				fenics.solve(self.form_handler.state_eq_form==0, self.state, self.bcs)
			
			self.has_solution = True
			self.number_of_solves += 1

		return self.state
