"""
Created on 24/02/2020, 09.26

@author: blauths
"""

import fenics



class GradientProblem:
	"""A class representing a gradient problem for optimal control problems

	"""
	
	def __init__(self, form_handler, state_problem, adjoint_problem):
		"""Initializes the gradient problem
		
		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			the FormHandler object of the optimization problem
		state_problem : cestrel._pde_problems.StateProblem
			the StateProblem object used to solve the state equations
		adjoint_problem : cestrel._pde_problems.AdjointProblem
			the AdjointProblem used to solve the adjoint equations
		"""
		
		self.form_handler = form_handler
		self.state_problem = state_problem
		self.adjoint_problem = adjoint_problem
		
		self.gradients = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.config = self.form_handler.config

		self.has_solution = False


	
	def solve(self):
		"""Solves the Riesz projection problem to obtain the gradient of the (reduced) cost functional
		
		Returns
		-------
		gradients : list[dolfin.function.function.Function]
			the gradient of the cost functional
		"""
		
		self.state_problem.solve()
		self.adjoint_problem.solve()
		
		if not self.has_solution:
			for i in range(self.form_handler.control_dim):
				b = fenics.as_backend_type(fenics.assemble(self.form_handler.gradient_forms_rhs[i])).vec()
				x = self.gradients[i].vector().vec()
				self.form_handler.ksps[i].solve(b, x)

				if self.form_handler.ksps[i].getConvergedReason() < 0:
					raise Exception('Krylov solver did not converge. Reason: ' + str(self.form_handler.ksps[i].getConvergedReason()))

			self.has_solution = True

			self.gradient_norm_squared = self.form_handler.scalar_product(self.gradients, self.gradients)

		return self.gradients
