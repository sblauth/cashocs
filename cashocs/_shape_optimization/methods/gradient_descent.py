# Copyright (C) 2020 Sebastian Blauth
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

"""
Created on 15/06/2020, 08.01

@author: blauths
"""

import numpy as np

from ..._exceptions import NotConvergedError
from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm



class GradientDescent(ShapeOptimizationAlgorithm):
	"""A gradient descent method for shape optimization

	"""

	def __init__(self, optimization_problem):
		"""A gradient descent method to solve the optimization problem

		Parameters
		----------
		optimization_problem : cashocs.optimization.optimization_problem.OptimalControlProblem
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

		try:
			self.iteration = self.optimization_problem.temp_dict['OptimizationRoutine'].get('iteration_counter', 0)
			self.gradient_norm_initial = self.optimization_problem.temp_dict['OptimizationRoutine'].get('gradient_norm_initial', 0.0)
		except TypeError:
			self.iteration = 0
			self.gradient_norm_initial = 0.0
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
					raise NotConvergedError('Armijo rule failed.')

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				self.print_results()
				if self.soft_exit:
					print('')
					print('Maximum number of iterations exceeded.')
					break
				else:
					raise NotConvergedError('Maximum number of iterations exceeded.')
