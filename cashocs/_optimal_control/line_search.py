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

"""Line search for optimal control problems.

"""

import fenics
import numpy as np


from .._loggers import error
from ..utils import _optimization_algorithm_configuration



class ArmijoLineSearch:
	"""An Armijo-based line search for optimal control

	Implements an Armijo line search for the solution of control problems.
	The exact behavior can be controlled via the config file.
	"""

	def __init__(self, optimization_algorithm):
		"""Initializes the line search object

		Parameters
		----------
		optimization_algorithm : cashocs._optimal_control.optimization_algorithm.OptimizationAlgorithm
			the corresponding optimization algorihm
		"""

		self.optimization_algorithm = optimization_algorithm
		self.config = self.optimization_algorithm.config
		self.optimization_problem = self.optimization_algorithm.optimization_problem
		self.form_handler = self.optimization_problem.form_handler

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'initial_stepsize', fallback=1.0)
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo', fallback=1e-4)
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo', fallback=2.0)
		self.armijo_stepsize_initial = self.stepsize

		self.cost_functional = self.optimization_problem.reduced_cost_functional
		self.projected_difference = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.controls = self.optimization_algorithm.controls
		self.controls_temp = self.optimization_algorithm.controls_temp
		self.gradients = self.optimization_algorithm.gradients

		self.is_newton_like = (_optimization_algorithm_configuration(self.config) == 'lbfgs')
		self.is_newton = (_optimization_algorithm_configuration(self.config) == 'newton')
		self.is_steepest_descent = (_optimization_algorithm_configuration(self.config) == 'gradient_descent')
		if self.is_newton:
			self.stepsize = 1.0



	def decrease_measure(self):
		"""Computes the measure of decrease needed for the Armijo test

		Returns
		-------
		float
			the decrease measure for the Armijo test
		"""

		for j in range(self.form_handler.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.controls_temp[j].vector()[:]

		return self.form_handler.scalar_product(self.gradients, self.projected_difference)



	def search(self, search_directions, has_curvature_info):
		"""Does a line search with the Armijo rule.

		Performs the line search along the entered search direction and adapts
		the step size if curvature information is contained in the search direction.

		Parameters
		----------
		search_directions : list[dolfin.function.function.Function]
			the current search direction computed by the optimization algorithm
		has_curvature_info : bool
			boolean flag, indicating whether the search direction is (actually) computed by
			a BFGS or Newton method

		Returns
		-------
		None
		"""

		self.search_direction_inf = np.max([np.max(np.abs(search_directions[i].vector()[:])) for i in range(len(self.gradients))])
		self.optimization_algorithm.objective_value = self.cost_functional.evaluate()

		if has_curvature_info:
			self.stepsize = 1.0

		self.optimization_algorithm.print_results()

		for j in range(self.form_handler.control_dim):
			self.controls_temp[j].vector()[:] = self.controls[j].vector()[:]

		while True:
			if self.stepsize*self.search_direction_inf <= 1e-8:
				error('Stepsize too small.')
				self.optimization_algorithm.line_search_broken = True
				for j in range(self.form_handler.control_dim):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				break
			elif not self.is_newton_like and not self.is_newton and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
				self.optimization_algorithm.line_search_broken = True
				error('Stepsize too small.')
				for j in range(self.form_handler.control_dim):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				break

			for j in range(len(self.controls)):
				self.controls[j].vector()[:] += self.stepsize*search_directions[j].vector()[:]

			self.form_handler.project_to_admissible_set(self.controls)

			self.optimization_algorithm.state_problem.has_solution = False
			self.objective_step = self.cost_functional.evaluate()

			# self.project_direction_active(search_directions)
			# meas = -self.epsilon_armijo*self.stepsize*self.form_handler.scalar_product(self.gradients, self.directions)

			if self.objective_step < self.optimization_algorithm.objective_value + self.epsilon_armijo*self.decrease_measure():
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
