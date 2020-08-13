"""
Created on 24/02/2020, 08.44

@author: blauths
"""

import fenics



class ReducedCostFunctional:
	"""The reduced cost functional for the optimization problem

	A class that represents an reduced cost functional of an optimal control problem, which
	is used to evaluate it.
	"""
	
	def __init__(self, form_handler, state_problem):
		"""Initialize the reduced cost functional
		
		Parameters
		----------
		form_handler : caospy._forms.FormHandler
			the FormHandler object for the optimization problem
		state_problem : caospy._pde_problems.StateProblem
			the StateProblem object corresponding to the state system
		"""
		
		self.form_handler = form_handler
		self.state_problem = state_problem
	


	def evaluate(self):
		"""Evaluates the reduced cost functional.

		First solves the state system, so that the state variables are up-to-date,
		and then evaluates the reduced cost functional by assembling the corresponding
		UFL form.

		Returns
		-------
		float
			the value of the reduced cost functional
		"""
		
		self.state_problem.solve()
		
		return fenics.assemble(self.form_handler.cost_functional_form)
