# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Line search for inner PDAS solvers.

"""

import numpy as np



class UnconstrainedLineSearch:
	"""Armijo line search for unconstrained optimization problems

	"""

	def __init__(self, optimization_algorithm):
		"""Initializes the line search

		Parameters
		----------
		config : configparser.ConfigParser
			the config file for the problem
		optimization_algorithm : cashocs._optimal_control.OptimizationAlgorithm
			the corresponding optimization algorithm
		"""

		self.optimization_algorithm = optimization_algorithm
		self.config = self.optimization_algorithm.config
		self.optimization_problem = self.optimization_algorithm.optimization_problem
		self.form_handler = self.optimization_problem.form_handler

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'initial_stepsize', fallback=1.0)
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo', fallback=1e-4)
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo', fallback=2.0)
		self.armijo_stepsize_initial = self.stepsize

		self.controls_temp = self.optimization_algorithm.controls_temp
		self.cost_functional = self.optimization_problem.reduced_cost_functional

		self.controls = self.optimization_algorithm.controls
		self.gradients = self.optimization_algorithm.gradients

		inner_pdas = self.config.get('AlgoPDAS', 'inner_pdas')
		self.is_newton_like = inner_pdas in ['lbfgs', 'bfgs']
		self.is_newton = inner_pdas in ['newton']
		self.is_steepest_descent = inner_pdas in ['gradient_descent', 'gd']
		if self.is_newton:
			self.stepsize = 1.0



	def decrease_measure(self, search_directions):
		"""Computes the decrease measure for the Armijo rule

		Parameters
		----------
		search_directions : list[dolfin.function.function.Function]
			the search direction computed by optimization_algorithm

		Returns
		-------
		float
			the decrease measure for the Armijo rule
		"""

		return self.stepsize*self.form_handler.scalar_product(self.gradients, search_directions)



	def search(self, search_directions):
		"""Performs an Armijo line search

		Parameters
		----------
		search_directions : list[dolfin.function.function.Function]
			the search direction computed by the optimization_algorithm

		Returns
		-------
		None
		"""

		self.search_direction_inf = np.max([np.max(np.abs(search_directions[i].vector()[:])) for i in range(len(self.gradients))])
		self.optimization_algorithm.objective_value = self.cost_functional.evaluate()

		# self.optimization_algorithm.print_results()

		for j in range(self.form_handler.control_dim):
			self.controls_temp[j].vector()[:] = self.controls[j].vector()[:]

		while True:
			if self.stepsize*self.search_direction_inf <= 1e-8:
				self.optimization_algorithm.line_search_broken = True
				for j in range(self.form_handler.control_dim):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				break
			elif not self.is_newton_like and not self.is_newton and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
				self.optimization_algorithm.line_search_broken = True
				for j in range(self.form_handler.control_dim):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				break

			for j in range(len(self.controls)):
				self.controls[j].vector()[:] += self.stepsize*search_directions[j].vector()[:]


			self.optimization_algorithm.state_problem.has_solution = False
			self.objective_step = self.cost_functional.evaluate()

			if self.objective_step < self.optimization_algorithm.objective_value + self.epsilon_armijo*self.decrease_measure(search_directions):
				if self.optimization_algorithm.iteration == 0:
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
