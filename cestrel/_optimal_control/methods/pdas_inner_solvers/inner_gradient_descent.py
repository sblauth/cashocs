"""
Created on 01.04.20, 11:43

@author: sebastian
"""

import fenics
import numpy as np
from ...optimization_algorithm import OptimizationAlgorithm
from .unconstrained_line_search import UnconstrainedLineSearch
from ...._exceptions import NotConvergedError



class InnerGradientDescent(OptimizationAlgorithm):
	"""A unconstrained gradient descent method.

	"""

	def __init__(self, optimization_problem):
		"""Initializes the gradient descent method.

		Parameters
		----------
		optimization_problem : cestrel.OptimalControlProblem
			the corresponding optimal control problem to be solved
		"""

		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = UnconstrainedLineSearch(self)
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations_inner_pdas', fallback=50)
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'pdas_inner_tolerance', fallback=1e-2)
		self.reduced_gradient = [fenics.Function(self.optimization_problem.control_spaces[j]) for j in range(len(self.controls))]
		self.first_iteration = True
		self.first_gradient_norm = 1.0


	def run(self, idx_active):
		"""Solves the inner PDAS optimization problem with the gradient descent method

		Parameters
		----------
		idx_active : list[numpy.ndarray]
			list of indices for the active set

		Returns
		-------
		None
		"""

		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		self.line_search.stepsize = 1.0

		while True:

			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()

			for j in range(len(self.controls)):
				self.reduced_gradient[j].vector()[:] = self.gradients[j].vector()[:]
				self.reduced_gradient[j].vector()[idx_active[j]] = 0.0

			self.gradient_norm = np.sqrt(self.form_handler.scalar_product(self.reduced_gradient, self.reduced_gradient))

			if self.iteration==0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0.0:
					self.print_results()
					break
				if self.first_iteration:
					self.first_gradient_norm = self.gradient_norm_initial
					self.first_iteration = False

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial or self.relative_norm*self.gradient_norm_initial/self.first_gradient_norm <= self.tolerance/2:
				self.print_results()
				break

			for i in range(len(self.controls)):
				self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
				self.search_directions[i].vector()[idx_active[i]] = 0

			self.line_search.search(self.search_directions)
			if self.line_search_broken:
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
