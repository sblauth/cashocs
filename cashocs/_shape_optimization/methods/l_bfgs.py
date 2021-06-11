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

"""Limited memory BFGS methods for shape optimization.

"""

from _collections import deque

import fenics
import numpy as np

from ..._loggers import debug
from ..._shape_optimization import ArmijoLineSearch, ShapeOptimizationAlgorithm



class LBFGS(ShapeOptimizationAlgorithm):

	def __init__(self, optimization_problem):
		"""Implements the L-BFGS method for solving the optimization problem

		Parameters
		----------
		optimization_problem : cashocs.shape_optimization.shape_optimization_problem.ShapeOptimizationProblem
			instance of the OptimalControlProblem (user defined)
		"""

		ShapeOptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.temp = fenics.Function(self.form_handler.deformation_space)

		self.bfgs_memory_size = self.config.getint('AlgoLBFGS', 'bfgs_memory_size', fallback=5)
		self.use_bfgs_scaling = self.config.getboolean('AlgoLBFGS', 'use_bfgs_scaling', fallback=True)

		self.has_curvature_info = False

		if self.bfgs_memory_size > 0:
			self.history_s = deque()
			self.history_y = deque()
			self.history_rho = deque()
			self.gradient_prev = fenics.Function(self.form_handler.deformation_space)
			self.y_k = fenics.Function(self.form_handler.deformation_space)
			self.s_k = fenics.Function(self.form_handler.deformation_space)



	def compute_search_direction(self, grad):
		"""Computes the search direction for the BFGS method with the so-called double loop

		Parameters
		----------
		grad : dolfin.function.function.Function
			the current gradient

		Returns
		-------
		self.search_direction : dolfin.function.function.Function
			a function corresponding to the current / next search direction

		"""

		if self.bfgs_memory_size > 0 and len(self.history_s) > 0:
			history_alpha = deque()
			self.search_direction.vector()[:] = grad.vector()[:]

			for i, _ in enumerate(self.history_s):
				alpha = self.history_rho[i]*self.form_handler.scalar_product(self.history_s[i], self.search_direction)
				history_alpha.append(alpha)
				self.search_direction.vector()[:] -= alpha * self.history_y[i].vector()[:]

			if self.use_bfgs_scaling and self.iteration > 0:
				factor = self.form_handler.scalar_product(self.history_y[0], self.history_s[0]) / self.form_handler.scalar_product(self.history_y[0], self.history_y[0])
			else:
				factor = 1.0

			self.search_direction.vector()[:] *= factor

			for i, _ in enumerate(self.history_s):
				beta = self.history_rho[-1 - i]*self.form_handler.scalar_product(self.history_y[-1 - i], self.search_direction)

				self.search_direction.vector()[:] += self.history_s[-1 - i].vector()[:] * (history_alpha[-1 -i] - beta)

			self.search_direction.vector()[:] *= -1

		else:
			self.search_direction.vector()[:] = -grad.vector()[:]

		return self.search_direction



	def run(self):
		"""Performs the optimization via the limited memory BFGS method

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
		self.state_problem.has_solution = False

		self.adjoint_problem.has_solution = False
		self.shape_gradient_problem.has_solution = False
		self.shape_gradient_problem.solve()
		self.gradient_norm = np.sqrt(self.shape_gradient_problem.gradient_norm_squared)

		if self.gradient_norm_initial > 0.0:
			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
		else:
			self.gradient_norm_initial = self.gradient_norm
			self.relative_norm = 1.0


		while True:
			self.search_direction = self.compute_search_direction(self.gradient)

			self.directional_derivative = self.form_handler.scalar_product(self.search_direction, self.gradient)
			if self.directional_derivative > 0:
				debug('No descent direction found with L-BFGS')
				self.search_direction.vector()[:] = - self.gradient.vector()[:]

			self.line_search.search(self.search_direction, self.has_curvature_info)

			self.iteration += 1
			if self.nonconvergence():
				break

			if self.bfgs_memory_size > 0:
				self.gradient_prev.vector()[:] = self.gradient.vector()[:]

			self.adjoint_problem.has_solution = False
			self.shape_gradient_problem.has_solution = False
			self.shape_gradient_problem.solve()

			self.gradient_norm = np.sqrt(self.shape_gradient_problem.gradient_norm_squared)
			self.relative_norm = self.gradient_norm / self.gradient_norm_initial

			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				self.converged = True
				break

			if self.bfgs_memory_size > 0:
				self.y_k.vector()[:] = self.gradient.vector()[:] - self.gradient_prev.vector()[:]
				self.s_k.vector()[:] = self.stepsize*self.search_direction.vector()[:]
				
				self.history_y.appendleft(self.y_k.copy(True))
				self.history_s.appendleft(self.s_k.copy(True))

				self.curvature_condition = self.form_handler.scalar_product(self.y_k, self.s_k)

				# if self.curvature_condition <= 1e-14:
				if self.curvature_condition <= 0.0:
				# if self.curvature_condition / self.form_handler.scalar_product(self.s_k, self.s_k) < 1e-7 * self.gradient_problem.return_norm_squared():
					self.has_curvature_info = False

					self.history_s = deque()
					self.history_y = deque()
					self.history_rho = deque()

				else:
					self.has_curvature_info = True
					rho = 1/self.curvature_condition
					self.history_rho.appendleft(rho)

				if len(self.history_s) > self.bfgs_memory_size:
					self.history_s.pop()
					self.history_y.pop()
					self.history_rho.pop()
