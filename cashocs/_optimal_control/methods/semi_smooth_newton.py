"""
Created on 06/03/2020, 10.21

@author: blauths
"""

import numpy as np
from ..._optimal_control import OptimizationAlgorithm, ArmijoLineSearch
from ..._exceptions import NotConvergedError



class SemiSmoothNewton(OptimizationAlgorithm):
	"""A semi-smooth Newton method.

	"""

	def __init__(self, optimization_problem):
		"""Initializes the semi-smooth Newton method.

		Parameters
		----------
		optimization_problem : cashocs.OptimalControlProblem
			the OptimalControlProblem object
		"""

		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo', fallback=1e-4)
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo', fallback=2)
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose', fallback=True)
		self.stepsize = 1.0
		self.armijo_stepsize_initial = self.stepsize

		self.armijo_broken = False



	def run(self):
		"""Performs the optimization via the semi-smooth Newton method

		Returns
		-------
		None
		"""

		self.iteration = 0
		self.relative_norm = 1.0


		while True:
			self.state_problem.has_solution = False
			self.objective_value = self.cost_functional.evaluate()

			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			self.gradient_norm = np.sqrt(self.optimization_problem._stationary_measure_squared())
			if self.iteration == 0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0.0:
					self.objective_value = self.cost_functional.evaluate()
					self.print_results()
					break

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				if self.iteration == 0:
					self.objective_value = self.cost_functional.evaluate()
				self.print_results()
				break

			self.print_results()

			self.search_directions, self.delta_mu = self.optimization_problem.semi_smooth_hessian.newton_solve()
			self.directional_derivative = self.form_handler.scalar_product(self.search_directions, self.gradients)

			self.idx_inactive = self.optimization_problem.semi_smooth_hessian.idx_inactive
			self.idx_active = self.optimization_problem.semi_smooth_hessian.idx_active
			self.mu = self.optimization_problem.semi_smooth_hessian.mu

			for j in range(len(self.controls)):
				self.controls[j].vector()[:] += self.search_directions[j].vector()[:]
				self.mu[j].vector()[:] += self.delta_mu[j].vector()[:]
				self.mu[j].vector()[self.optimization_problem.form_handler.idx_inactive[j]] = 0

			if self.armijo_broken:
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
