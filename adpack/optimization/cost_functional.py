"""
Created on 24/02/2020, 08.44

@author: blauths
"""

import fenics
import numpy as np



class CostFunctionalForm:
	def __init__(self, state_form):
		_, _, self.state, self.control, _ = state_form.return_data()
		self.discretization = state_form.discretization
		
		self.form = fenics.Constant(0)*self.discretization.dx
	




class CostFunctional:
	def __init__(self, form_handler, state_problem):
		self.form_handler = form_handler
		self.state_problem = state_problem
	
	
	def compute(self):
		self.state_problem.solve()
		
		return fenics.assemble(self.form_handler.cost_functional_form)
