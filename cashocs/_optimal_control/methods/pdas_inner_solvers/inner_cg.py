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

"""Nonlinear CG methods for the solution of PDAS problems.

"""

import fenics
import numpy as np

from .unconstrained_line_search import UnconstrainedLineSearch
from ...optimization_algorithm import OptimizationAlgorithm
from ...._exceptions import ConfigError, NotConvergedError



class InnerCG(OptimizationAlgorithm):
	"""Unconstrained nonlinear conjugate gradient method.

	"""

	def __init__(self, optimization_problem):
		"""Initializes the nonlinear CG method.

		Parameters
		----------
		optimization_problem : cashocs.OptimalControlProblem
			the corresponding optimization problem
		"""

		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = UnconstrainedLineSearch(self)

		self.maximum_iterations = self.config.getint('AlgoPDAS', 'maximum_iterations_inner_pdas', fallback=50)
		self.tolerance = self.config.getfloat('AlgoPDAS', 'pdas_inner_tolerance', fallback=1e-2)
		self.reduced_gradient = [fenics.Function(self.optimization_problem.control_spaces[j]) for j in range(len(self.controls))]
		self.first_iteration = True
		self.first_gradient_norm = 1.0

		self.gradients_prev = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.differences = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.temp_HZ = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.armijo_broken = False

		self.cg_method = self.config.get('AlgoCG', 'cg_method', fallback='FR')
		self.cg_periodic_restart = self.config.getboolean('AlgoCG', 'cg_periodic_restart', fallback=False)
		self.cg_periodic_its = self.config.getint('AlgoCG', 'cg_periodic_its', fallback=10)
		self.cg_relative_restart = self.config.getboolean('AlgoCG', 'cg_relative_restart', fallback=False)
		self.cg_restart_tol = self.config.getfloat('AlgoCG', 'cg_restart_tol', fallback=0.25)
		
		self.pdas_solver = True



	def run(self, idx_active):
		"""Solves the inner optimization problem with the nonlinear CG method

		Parameters
		----------
		idx_active : list[numpy.ndarray]
			list of the indices corresponding to the active set
		Returns
		-------
		None
		"""

		self.iteration = 0
		self.memory = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False
		for i in range(len(self.gradients)):
			self.gradients[i].vector()[:] = 1.0
			self.reduced_gradient[i].vector()[:] = 1.0

		while True:

			for i in range(self.form_handler.control_dim):
				self.gradients_prev[i].vector()[:] = self.reduced_gradient[i].vector()[:]

			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()

			for j in range(len(self.controls)):
				self.reduced_gradient[j].vector()[:] = self.gradients[j].vector()[:]
				self.reduced_gradient[j].vector()[idx_active[j]] = 0.0

			self.gradient_norm = np.sqrt(self.form_handler.scalar_product(self.reduced_gradient, self.reduced_gradient))

			if self.iteration > 0:
				if self.cg_method=='FR':
					self.beta_numerator = self.form_handler.scalar_product(self.reduced_gradient, self.reduced_gradient)
					self.beta_denominator = self.form_handler.scalar_product(self.gradients_prev, self.gradients_prev)
					self.beta = self.beta_numerator/self.beta_denominator

				elif self.cg_method=='PR':
					for i in range(self.form_handler.control_dim):
						self.differences[i].vector()[:] = self.reduced_gradient[i].vector()[:] - self.gradients_prev[i].vector()[:]

					self.beta_numerator = self.form_handler.scalar_product(self.reduced_gradient, self.differences)
					self.beta_denominator = self.form_handler.scalar_product(self.gradients_prev, self.gradients_prev)
					self.beta = self.beta_numerator/self.beta_denominator


				elif self.cg_method=='HS':
					for i in range(self.form_handler.control_dim):
						self.differences[i].vector()[:] = self.reduced_gradient[i].vector()[:] - self.gradients_prev[i].vector()[:]

					self.beta_numerator = self.form_handler.scalar_product(self.reduced_gradient, self.differences)
					self.beta_denominator = self.form_handler.scalar_product(self.differences, self.search_directions)
					self.beta = self.beta_numerator/self.beta_denominator

				elif self.cg_method=='DY':
					for i in range(self.form_handler.control_dim):
						self.differences[i].vector()[:] = self.reduced_gradient[i].vector()[:] - self.gradients_prev[i].vector()[:]

					self.beta_numerator = self.form_handler.scalar_product(self.reduced_gradient, self.reduced_gradient)
					self.beta_denominator = self.form_handler.scalar_product(self.search_directions, self.differences)
					self.beta = self.beta_numerator/self.beta_denominator

				elif self.cg_method=='HZ':
					for i in range(self.form_handler.control_dim):
						self.differences[i].vector()[:] = self.reduced_gradient[i].vector()[:] - self.gradients_prev[i].vector()[:]

					dy = self.form_handler.scalar_product(self.search_directions, self.differences)
					y2 = self.form_handler.scalar_product(self.differences, self.differences)

					for i in range(self.form_handler.control_dim):
						self.differences[i].vector()[:] = self.differences[i].vector()[:] - 2*y2/dy*self.search_directions[i].vector()[:]

					self.beta = self.form_handler.scalar_product(self.differences, self.reduced_gradient) / dy

				else:
					raise ConfigError('AlgoCG', 'cg_method', 'Not a valid input. Choose either \'FR\' (Fletcher Reeves), \'PR\' (Polak Ribiere), '
									  '\'HS\' (Hestenes Stiefel), \'DY\' (Dai Yuan), or \'HZ\' (Hager Zhang).')

			if self.iteration==0:
				self.gradient_norm_initial = self.gradient_norm
				if self.first_iteration:
					self.first_gradient_norm = self.gradient_norm_initial
					self.first_iteration = False
				self.beta = 0.0

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.tolerance*self.gradient_norm_initial or self.relative_norm*self.gradient_norm_initial/self.first_gradient_norm <= self.tolerance/2:
				# self.print_results()
				break

			for i in range(self.form_handler.control_dim):
				self.search_directions[i].vector()[:] = -self.reduced_gradient[i].vector()[:] + self.beta*self.search_directions[i].vector()[:]

			if self.cg_periodic_restart:
				if self.memory < self.cg_periodic_its:
					self.memory += 1
				else:
					for i in range(self.form_handler.control_dim):
						self.search_directions[i].vector()[:] = -self.reduced_gradient[i].vector()[:]
					self.memory = 0

			if self.cg_relative_restart:
				if abs(self.form_handler.scalar_product(self.reduced_gradient, self.gradients_prev)) / pow(self.gradient_norm, 2) >= self.cg_restart_tol:
					for i in range(self.form_handler.control_dim):
						self.search_directions[i].vector()[:] = -self.reduced_gradient[i].vector()[:]

			self.directional_derivative = self.form_handler.scalar_product(self.reduced_gradient, self.search_directions)

			if self.directional_derivative >= 0:
				for i in range(self.form_handler.control_dim):
					self.search_directions[i].vector()[:] = -self.reduced_gradient[i].vector()[:]

			self.line_search.search(self.search_directions)
			if self.armijo_broken:
				if self.soft_exit:
					if self.verbose:
						print('Armijo rule failed.')
					break
				else:
					raise NotConvergedError('Armijo line search')

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				if self.soft_exit:
					if self.verbose:
						print('Maximum number of iterations exceeded.')
					break
				else:
					raise NotConvergedError('nonlinear CG method for the primal dual active set method', 'Maximum number of iterations were exceeded.')
