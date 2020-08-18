"""
Created on 24/02/2020, 08.44

@author: blauths
"""

import fenics



class ReducedShapeCostFunctional:
	"""Reduced cost functional for a shape optimization problem

	"""

	def __init__(self, shape_form_handler, state_problem):
		"""Initializes the reduced cost functional
		
		Parameters
		----------
		shape_form_handler : descendal._forms.ShapeFormHandler
			the ControlFormHandler object for the optimization problem
		state_problem : descendal._pde_problems.StateProblem
			the StateProblem object corresponding to the state system
		"""
		
		self.shape_form_handler = shape_form_handler
		self.state_problem = state_problem
		self.regularization = self.shape_form_handler.regularization



	def evaluate(self):
		"""Evaluates the reduced cost functional
		
		Returns
		-------
		float
			the value of the reduced cost functional at the current control

		"""
		
		self.state_problem.solve()
		# self.regularization.update_geometric_quantities()

		return fenics.assemble(self.shape_form_handler.cost_functional_form) + self.regularization.compute_objective()
