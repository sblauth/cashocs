"""
Created on 24/02/2020, 09.33

@author: blauths
"""

import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from ..line_search import ArmijoLineSearch



class GradientDescent(OptimizationAlgorithm):
	
	def __init__(self, optimization_problem):
		"""A gradient descent method to solve the optimization problem
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimizationProblem
			the OptimizationProblem object
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)


	
	def run(self):
		"""Performs the optimization via the gradient descent method
		
		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""
		
		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		while True:

			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			self.gradient_norm_squared = self.optimization_problem.stationary_measure_squared()

			if self.iteration == 0:
				self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)

			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			if self.relative_norm <= self.tolerance:
				self.converged = True
				break
			
			for i in range(len(self.controls)):
				self.controls_temp[i].vector()[:] = self.controls[i].vector()[:]
				self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]

			self.line_search.search(self.search_directions)
			if self.line_search_broken:
				print('Armijo rule failed')
				break

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				break

		if self.converged:
			self.print_results()

		print('')
		print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
			  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
		print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
			  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
		print('')
