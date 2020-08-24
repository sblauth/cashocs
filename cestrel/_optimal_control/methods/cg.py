"""
Created on 24/02/2020, 15.40

@author: blauths
"""

import fenics
import numpy as np
from ..._optimal_control import OptimizationAlgorithm, ArmijoLineSearch
from ..._exceptions import ConfigError, NotConvergedError



class CG(OptimizationAlgorithm):
	"""Nonlinear conjugate gradient method.

	"""

	def __init__(self, optimization_problem):
		"""Initializes the nonlinear cg method.

		Parameters
		----------
		optimization_problem : cestrel.OptimalControlProblem
			the OptimalControlProblem object
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.gradients_prev = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.differences = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.temp_HZ = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.cg_method = self.config.get('OptimizationRoutine', 'cg_method', fallback='FR')
		self.cg_periodic_restart = self.config.getboolean('OptimizationRoutine', 'cg_periodic_restart', fallback=False)
		self.cg_periodic_its = self.config.getint('OptimizationRoutine', 'cg_periodic_its', fallback=10)
		self.cg_relative_restart = self.config.getboolean('OptimizationRoutine', 'cg_relative_restart', fallback=False)
		self.cg_restart_tol = self.config.getfloat('OptimizationRoutine', 'cg_restart_tol', fallback=0.25)



	def project_direction(self, a):
		"""Restricts the search direction to the inactive set.

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			A function that shall be projected / restricted (will be overwritten)

		Returns
		-------
		None
		"""

		self.control_constraints = self.optimization_problem.control_constraints

		for j in range(self.form_handler.control_dim):
			idx = np.asarray(np.logical_or(np.logical_and(self.controls[j].vector()[:] <= self.control_constraints[j][0].vector()[:], a[j].vector()[:] < 0.0),
										   np.logical_and(self.controls[j].vector()[:] >= self.control_constraints[j][1].vector()[:], a[j].vector()[:] > 0.0))
										   ).nonzero()[0]

			a[j].vector()[idx] = 0.0

	
	
	def run(self):
		"""Performs the optimization via the nonlinear cg method
		
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

		while True:

			for i in range(self.form_handler.control_dim):
				self.gradients_prev[i].vector()[:] = self.gradients[i].vector()[:]

			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()

			self.gradient_norm = np.sqrt(self.optimization_problem._stationary_measure_squared())
			if self.iteration > 0:
				if self.cg_method == 'FR':
					self.beta_numerator = self.form_handler.scalar_product(self.gradients, self.gradients)
					self.beta_denominator = self.form_handler.scalar_product(self.gradients_prev, self.gradients_prev)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'PR':
					for i in range(len(self.gradients)):
						self.differences[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]

					self.beta_numerator = self.form_handler.scalar_product(self.gradients, self.differences)
					self.beta_denominator = self.form_handler.scalar_product(self.gradients_prev, self.gradients_prev)
					self.beta = self.beta_numerator / self.beta_denominator


				elif self.cg_method == 'HS':
					for i in range(len(self.gradients)):
						self.differences[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]

					self.beta_numerator = self.form_handler.scalar_product(self.gradients, self.differences)
					self.beta_denominator = self.form_handler.scalar_product(self.differences, self.search_directions)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'DY':
					for i in range(len(self.gradients)):
						self.differences[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]

					self.beta_numerator = self.form_handler.scalar_product(self.gradients, self.gradients)
					self.beta_denominator = self.form_handler.scalar_product(self.search_directions, self.differences)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'CD':
					self.beta_numerator = self.form_handler.scalar_product(self.gradients, self.gradients)
					self.beta_denominator = -self.form_handler.scalar_product(self.search_directions, self.gradients)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'HZ':
					for i in range(len(self.gradients)):
						self.differences[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]

					dy = self.form_handler.scalar_product(self.search_directions, self.differences)
					y2 = self.form_handler.scalar_product(self.differences, self.differences)

					for i in range(len(self.gradients)):
						self.differences[i].vector()[:] = self.differences[i].vector()[:] - 2*y2/dy*self.search_directions[i].vector()[:]

					self.beta = self.form_handler.scalar_product(self.differences, self.gradients) / dy

				else:
					raise ConfigError('Not a valid choice for OptimizationRoutine.cg_method. Choose either FR (Fletcher Reeves), PR (Polak Ribiere), HS (Hestenes Stiefel), DY (Dai Yuan), CD (Conjugate Descent) or HZ (Hager Zhang).')

			if self.iteration == 0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0:
					self.print_results()
					break
				self.beta = 0.0

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				self.print_results()
				break

			for i in range(self.form_handler.control_dim):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:] + self.beta*self.search_directions[i].vector()[:]
			if self.cg_periodic_restart:
				if self.memory < self.cg_periodic_its:
					self.memory += 1
				else:
					for i in range(len(self.gradients)):
						self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
					self.memory = 0
			if self.cg_relative_restart:
				if abs(self.form_handler.scalar_product(self.gradients, self.gradients_prev)) / pow(self.gradient_norm, 2) >= self.cg_restart_tol:
					for i in range(len(self.gradients)):
						self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]

			self.project_direction(self.search_directions)
			self.directional_derivative = self.form_handler.scalar_product(self.gradients, self.search_directions)
			
			if self.directional_derivative >= 0:
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]

			self.line_search.search(self.search_directions, self.has_curvature_info)
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
