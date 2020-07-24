"""
Created on 05/03/2020, 08.33

@author: blauths
"""

import fenics
import numpy as np



class ArmijoLineSearch:

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

		self.cost_functional = self.optimization_problem.reduced_cost_functional
		self.projected_difference = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.controls = self.optimization_algorithm.controls
		self.controls_temp = self.optimization_algorithm.controls_temp
		self.gradients = self.optimization_algorithm.gradients

		self.is_newton_like = self.config.get('OptimizationRoutine', 'algorithm') in ['lbfgs']
		self.is_newton = self.config.get('OptimizationRoutine', 'algorithm') in ['newton', 'semi_smooth_newton']
		self.is_steepest_descent = self.config.get('OptimizationRoutine', 'algorithm') in ['gradient_descent']
		if self.is_newton:
			self.stepsize = 1.0



	def decrease_measure(self):
		"""Computes the measure of decrease needed for the Armijo test

		Returns
		-------
		 : float
		"""
		### TODO: Check this
		if self.is_steepest_descent:
			return self.stepsize*self.optimization_problem.stationary_measure_squared()

		else:
			for j in range(self.form_handler.control_dim):
				self.projected_difference[j].vector()[:] = self.controls_temp[j].vector()[:] - self.controls[j].vector()[:]

			return self.form_handler.scalar_product(self.gradients, self.projected_difference)



	def search(self, search_directions, has_curvature_info):
		"""Performs the line search along the entered search direction and will adapt step if curvature information is contained in the search direction

		Parameters
		----------
		search_directions : list[dolfin.function.function.Function]
			The current search direction computed by the algorithms
		has_curvature_info : bool
			True if the step is (actually) computed via L-BFGS or Newton

		Returns
		-------

		"""

		self.search_direction_inf = np.max([np.max(np.abs(search_directions[i].vector()[:])) for i in range(len(self.gradients))])
		self.optimization_algorithm.objective_value = self.cost_functional.compute()

		if has_curvature_info:
			self.stepsize = 1.0

		self.optimization_algorithm.print_results()

		for j in range(self.form_handler.control_dim):
			self.controls_temp[j].vector()[:] = self.controls[j].vector()[:]

		while True:
			if self.stepsize*self.search_direction_inf <= 1e-8:
				print('\nStepsize too small.')
				self.optimization_algorithm.line_search_broken = True
				for j in range(self.form_handler.control_dim):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				break
			elif not self.is_newton_like and not self.is_newton and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
				self.optimization_algorithm.line_search_broken = True
				print('\nStepsize too small.')
				for j in range(self.form_handler.control_dim):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				break

			for j in range(len(self.controls)):
				self.controls[j].vector()[:] += self.stepsize*search_directions[j].vector()[:]

			self.form_handler.project(self.controls)

			self.optimization_algorithm.state_problem.has_solution = False
			self.objective_step = self.cost_functional.compute()

			if self.objective_step < self.optimization_algorithm.objective_value - self.epsilon_armijo*self.decrease_measure():
				if self.optimization_algorithm.iteration == 0:
					self.armijo_stepsize_initial = self.stepsize
				break

			else:
				self.stepsize /= self.beta_armijo
				for i in range(len(self.controls)):
					self.controls[i].vector()[:] = self.controls_temp[i].vector()[:]

		if not self.optimization_algorithm.line_search_broken:
			self.optimization_algorithm.stepsize = self.stepsize
			self.optimization_algorithm.objective_value = self.objective_step

		if not has_curvature_info:
			self.stepsize *= self.beta_armijo
