"""
Created on 15/06/2020, 08.01

@author: blauths
"""

import numpy as np
from ..._shape_optimization import ShapeOptimizationAlgorithm, ArmijoLineSearch




class GradientDescent(ShapeOptimizationAlgorithm):
	"""A gradient descent method for shape optimization

	"""

	def __init__(self, optimization_problem):
		"""A gradient descent method to solve the optimization problem

		Parameters
		----------
		optimization_problem : descendal.optimization.optimization_problem.OptimalControlProblem
			the OptimalControlProblem object
		"""

		ShapeOptimizationAlgorithm.__init__(self, optimization_problem)

		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose', fallback=True)
		self.line_search = ArmijoLineSearch(self)
		self.has_curvature_info = False



	def run(self):
		"""Performs the optimization via the gradient descent method

		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""

		self.iteration = self.config.getint('OptimizationRoutine', 'iteration_counter', fallback=0)
		self.gradient_norm_initial = self.config.getfloat('OptimizationRoutine', 'gradient_norm_initial', fallback=0.0)
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		while True:

			self.adjoint_problem.has_solution = False
			self.shape_gradient_problem.has_solution = False
			self.shape_gradient_problem.solve()
			self.gradient_norm = np.sqrt(self.shape_gradient_problem.gradient_norm_squared)

			if self.iteration == 0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0:
					self.print_results()
					break

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				self.print_results()
				break

			self.search_direction.vector()[:] = - self.gradient.vector()[:]

			self.line_search.search(self.search_direction, self.has_curvature_info)
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
