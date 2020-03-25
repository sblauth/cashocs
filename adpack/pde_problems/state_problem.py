"""
Created on 24/02/2020, 09.19

@author: blauths
"""

import fenics
import numpy as np
from phdutils.nonlinear_solvers import NewtonSolver



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
		self.bcs_list = self.form_handler.bcs_list
		self.states = self.form_handler.states
		self.controls = self.form_handler.controls
		
		self.number_of_solves = 0
		self.has_solution = False
	
	
	
	def solve(self):
		"""Solve command for the state equation. Solution is "stored" in self.state (corresponding to user input)
		
		Returns
		-------
		self.states : List[dolfin.function.function.Function]
			the function that corresponds to the solution of the state system

		"""
		
		if not self.has_solution:
			if self.config.getboolean('StateEquation', 'is_linear'):
				for i in range(self.form_handler.state_dim):
					a, L = fenics.system(self.form_handler.state_eq_forms[i])
					fenics.solve(a==L, self.states[i], self.bcs_list[i])
				
			else:
				for i in range(self.form_handler.state_dim):
					# fenics.solve(self.form_handler.state_eq_forms[i]==0, self.states[i], self.bcs_list[i])
					self.states[i] = NewtonSolver(self.form_handler.state_eq_forms[i], self.states[i], self.bcs_list[i], rtol=1e-9, atol=1e-7, damped=True, verbose=False)

			self.has_solution = True
			self.number_of_solves += 1

		return self.states
