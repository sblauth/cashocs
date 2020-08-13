"""
Created on 24/02/2020, 09.33

@author: blauths
"""

import numpy as np
from ..._optimal_control import OptimizationAlgorithm, ArmijoLineSearch




class GradientDescent(OptimizationAlgorithm):
	"""A gradient descent method

	"""

	def __init__(self, optimization_problem):
		"""Initializes the method.
		
		Parameters
		----------
		optimization_problem : caospy.OptimalControlProblem
			the OptimalControlProblem object
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.line_search = ArmijoLineSearch(self)
		self.has_curvature_info = False


	
	def run(self):
		"""Performs the optimization via the gradient descent method
		
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
			self.gradient_norm = np.sqrt(self.optimization_problem.stationary_measure_squared())

			if self.iteration == 0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0:
					self.print_results()
					break

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				self.print_results()
				break
			
			for i in range(len(self.controls)):
				self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]

			self.line_search.search(self.search_directions, self.has_curvature_info)
			if self.line_search_broken:
				if self.soft_exit:
					print('Armijo rule failed.')
					break
				else:
					raise SystemExit('Armijo rule failed.')

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				self.print_results()
				if self.soft_exit:
					print('')
					print('Maximum number of iterations exceeded.')
					break
				else:
					raise SystemExit('Maximum number of iterations exceeded.')

		if self.verbose:
			print('')
			print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
			print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
			print('')
