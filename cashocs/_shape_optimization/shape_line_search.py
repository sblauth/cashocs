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

"""Line search for shape optimization problems.

"""

import fenics
import numpy as np

from .._loggers import error
from ..utils import _optimization_algorithm_configuration



class ArmijoLineSearch:

	def __init__(self, optimization_algorithm):
		"""Initializes the line search

		Parameters
		----------
		optimization_algorithm : cashocs._shape_optimization.shape_optimization_algorithm.ShapeOptimizationAlgorithm
			the optimization problem of interest
		"""

		self.optimization_algorithm = optimization_algorithm
		self.config = self.optimization_algorithm.config
		self.optimization_problem = self.optimization_algorithm.optimization_problem
		self.form_handler = self.optimization_problem.form_handler
		self.mesh_handler = self.optimization_problem.mesh_handler
		self.deformation = fenics.Function(self.form_handler.deformation_space)

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'initial_stepsize', fallback=1.0)
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo', fallback=1e-4)
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo', fallback=2.0)
		self.armijo_stepsize_initial = self.stepsize

		self.cost_functional = self.optimization_problem.reduced_cost_functional

		self.gradient = self.optimization_algorithm.gradient

		self.algorithm = _optimization_algorithm_configuration(self.config)
		self.is_newton_like = (self.algorithm == 'lbfgs')
		self.is_newton = self.algorithm in ['newton']
		self.is_steepest_descent = (self.algorithm == 'gradient_descent')
		if self.is_newton:
			self.stepsize = 1.0



	def decrease_measure(self, search_direction):
		"""Computes the measure of decrease needed for the Armijo test

		Parameters
		----------
		search_direction : dolfin.function.function.Function
			The current search direction

		Returns
		-------
		float
			the decrease measure for the Armijo rule
		"""

		return self.stepsize*self.form_handler.scalar_product(self.gradient, search_direction)



	def search(self, search_direction, has_curvature_info):
		"""Performs the line search along the entered search direction

		Parameters
		----------
		search_direction : dolfin.function.function.Function
			The current search direction computed by the algorithms
		has_curvature_info : bool
			True if the step is (actually) computed via L-BFGS or Newton

		Returns
		-------
		None
		"""

		self.search_direction_inf = np.max(np.abs(search_direction.vector()[:]))
		self.optimization_algorithm.objective_value = self.cost_functional.evaluate()

		if has_curvature_info:
			self.stepsize = 1.0

		self.optimization_algorithm.print_results()

		num_decreases = self.mesh_handler.compute_decreases(search_direction, self.stepsize)
		self.stepsize /= pow(self.beta_armijo, num_decreases)

		while True:
			if self.optimization_algorithm.iteration >= self.optimization_algorithm.maximum_iterations:
				self.optimization_algorithm.remeshing_its = True
				break

			if self.stepsize*self.search_direction_inf <= 1e-8:
				error('Stepsize too small.')
				self.optimization_algorithm.line_search_broken = True
				break
			elif not self.is_newton_like and not self.is_newton and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
				error('Stepsize too small.')
				self.optimization_algorithm.line_search_broken = True
				break

			self.deformation.vector()[:] = self.stepsize*search_direction.vector()[:]

			if self.mesh_handler.move_mesh(self.deformation):
				if self.mesh_handler.current_mesh_quality < self.mesh_handler.mesh_quality_tol_lower:
					self.stepsize /= self.beta_armijo
					self.mesh_handler.revert_transformation()
					continue

				self.optimization_algorithm.state_problem.has_solution = False
				self.objective_step = self.cost_functional.evaluate()

				if self.objective_step < self.optimization_algorithm.objective_value + self.epsilon_armijo*self.decrease_measure(search_direction):

					if self.mesh_handler.current_mesh_quality < self.mesh_handler.mesh_quality_tol_upper:
						self.optimization_algorithm.requires_remeshing = True
						break

					if self.optimization_algorithm.iteration == 0:
						self.armijo_stepsize_initial = self.stepsize
					self.form_handler.update_scalar_product()
					break

				else:
					self.stepsize /= self.beta_armijo
					self.mesh_handler.revert_transformation()

			else:
				self.stepsize /= self.beta_armijo

		if not (self.optimization_algorithm.line_search_broken or self.optimization_algorithm.requires_remeshing or self.optimization_algorithm.remeshing_its):
				self.optimization_algorithm.stepsize = self.stepsize
				self.optimization_algorithm.objective_value = self.objective_step

		if not has_curvature_info:
			self.stepsize *= self.beta_armijo
