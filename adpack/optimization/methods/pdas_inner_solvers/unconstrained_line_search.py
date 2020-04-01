"""
Created on 01.04.20, 11:51

@author: sebastian
"""

import fenics
import numpy as np



class UnconstrainedLineSearch:

	def __init__(self, optimization_algorithm):
		"""

		Parameters
		----------
		config : configparser.ConfigParser
			the config file for the problem
		optimization_algorithm : adpack.optimization.optimization_algorithm.OptimizationAlgorithm
			the optimization problem of interest
		"""

		self.optimization_algorithm = optimization_algorithm
		self.config = self.optimization_algorithm.config
		self.optimization_problem = self.optimization_algorithm.optimization_problem
		self.form_handler = self.optimization_problem.form_handler

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.armijo_stepsize_initial = self.stepsize

		self.controls_temp = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.cost_functional = self.optimization_problem.reduced_cost_functional

		self.controls = self.optimization_algorithm.controls
		self.gradients = self.optimization_algorithm.gradients
		self.iteration = self.optimization_algorithm.iteration

		self.is_newton_like = self.config.get('OptimizationRoutine', 'algorithm') in ['lbfgs']
		self.is_newton = self.config.get('OptimizationRoutine', 'algorithm') in ['newton', 'semi_smooth_newton']
		self.is_steepest_descent = self.config.get('OptimizationRoutine', 'algorithm') in ['gradient_descent']
		if self.is_newton:
			self.stepsize = 1.0



	def decrease_measure(self, search_directions):
		return self.stepsize*self.form_handler.scalar_product(self.gradients, search_directions)



	def search(self, search_directions):

		self.search_direction_inf = np.max([np.max(np.abs(search_directions[i].vector()[:])) for i in range(len(self.gradients))])
		self.optimization_algorithm.objective_value = self.cost_functional.compute()

		# self.optimization_algorithm.print_results()

		for j in range(self.form_handler.control_dim):
			self.controls_temp[j].vector()[:] = self.controls[j].vector()[:]

		while True:
			if self.stepsize*self.search_direction_inf <= 1e-8:
				self.optimization_algorithm.line_search_broken = True
				break
			elif not self.is_newton_like and not self.is_newton and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
				self.optimization_algorithm.line_search_broken = True
				break

			for j in range(len(self.controls)):
				self.controls[j].vector()[:] += self.stepsize*search_directions[j].vector()[:]


			self.optimization_algorithm.state_problem.has_solution = False
			self.objective_step = self.cost_functional.compute()

			if self.objective_step < self.optimization_algorithm.objective_value + self.epsilon_armijo*self.decrease_measure(search_directions):
				if self.iteration == 0:
					self.armijo_stepsize_initial = self.stepsize
				break

			else:
				self.stepsize /= self.beta_armijo
				for i in range(len(self.controls)):
					self.controls[i].vector()[:] = self.controls_temp[i].vector()[:]

		self.optimization_algorithm.stepsize = self.stepsize
		self.optimization_algorithm.objective_value = self.objective_step
		if not self.is_newton_like and not self.is_newton:
			self.stepsize *= self.beta_armijo
		else:
			self.stepsize = 1.0
