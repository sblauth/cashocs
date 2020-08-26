"""
Created on 24/02/2020, 14.31

@author: blauths
"""

import numpy as np
from ..._optimal_control import OptimizationAlgorithm, ArmijoLineSearch
from ..._exceptions import NotConvergedError




class Newton(OptimizationAlgorithm):
	"""A truncated Newton method.

	"""

	def __init__(self, optimization_problem):
		"""Initializes the Newton method
		
		Parameters
		----------
		optimization_problem : cestrel.OptimalControlProblem
			the OptimalControlProblem object
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.stepsize = 1.0
		self.armijo_stepsize_initial = self.stepsize

		self.has_curvature_info = False

		self.armijo_broken = False



	def run(self):
		"""Performs the optimization via the truncated Newton method
		
		Returns
		-------
		None
		"""
		
		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		while True:
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			self.gradient_norm = np.sqrt(self.optimization_problem._stationary_measure_squared())
			if self.iteration == 0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0:
					self.objective_value = self.cost_functional.evaluate()
					self.print_results()
					break

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				if self.iteration == 0:
					self.objective_value = self.cost_functional.evaluate()
				self.print_results()
				break

			self.search_directions = self.optimization_problem.hessian_problem.newton_solve()
			self.directional_derivative = self.form_handler.scalar_product(self.search_directions, self.gradients)
			if self.directional_derivative > 0:
				self.has_curvature_info = False
				# print('No descent direction')
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
			else:
				self.has_curvature_info = True

			self.line_search.search(self.search_directions, self.has_curvature_info)

			if self.armijo_broken and self.has_curvature_info:
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
				self.has_curvature_info = False
				self.armijo_broken = False
				self.line_search.search(self.search_directions, self.has_curvature_info)

				if self.armijo_broken:
					if self.soft_exit:
						print('Armijo rule failed.')
						break
					else:
						raise NotConvergedError('Armijo rule failed.')

			elif self.armijo_broken and not self.has_curvature_info:
				if self.soft_exit:
					print('Armijo rule failed.')
					break
				else:
					raise NotConvergedError('Armijo rule failed.')

			self.iteration += 1

			if self.iteration >= self.maximum_iterations:
				self.print_results()
				if self.soft_exit:
					print('Maximum number of iterations exceeded.')
					break
				else:
					raise NotConvergedError('Maximum number of iterations exceeded.')
