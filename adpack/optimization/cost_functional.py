"""
Created on 24/02/2020, 08.44

@author: blauths
"""

import fenics



class ReducedCostFunctional:
	
	def __init__(self, form_handler, state_problem):
		"""An implementation of the reduced cost functional
		
		Parameters
		----------
		form_handler : adpack.forms.FormHandler
			the FormHandler object for the optimization problem
		state_problem : adpack.pde_problems.state_problem.StateProblem
			the StateProblem object corresponding to the state system
		"""
		
		self.form_handler = form_handler
		self.state_problem = state_problem
	


	def compute(self):
		"""Evaluates the reduced cost functional by first solving the state system and then evaluating the cost functional
		
		Returns
		-------
		 : float
			the value of the reduced cost functional at the current control

		"""
		
		self.state_problem.solve()
		
		return fenics.assemble(self.form_handler.cost_functional_form)
