"""
Created on 15/06/2020, 08.01

@author: blauths
"""

import fenics
import numpy as np
from ..._shape_optimization import ShapeOptimizationAlgorithm, ArmijoLineSearch



class CG(ShapeOptimizationAlgorithm):
	def __init__(self, optimization_problem):
		"""A nonlinear cg method to solve the optimization problem

		Additional parameters in the config file:
			cg_method : (one of) FR [Fletcher Reeves], PR [Polak Ribiere], HS [Hestenes Stiefel], DY [Dai-Yuan], CD [Conjugate Descent], HZ [Hager Zhang]

		Parameters
		----------
		optimization_problem : cestrel.shape_optimization.shape_optimization_problem.ShapeOptimizationProblem
			the OptimalControlProblem object
		"""

		ShapeOptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.gradient_prev = fenics.Function(self.shape_form_handler.deformation_space)
		self.difference = fenics.Function(self.shape_form_handler.deformation_space)
		self.temp_HZ = fenics.Function(self.shape_form_handler.deformation_space)

		self.cg_method = self.config.get('OptimizationRoutine', 'cg_method', fallback='FR')
		self.cg_periodic_restart = self.config.getboolean('OptimizationRoutine', 'cg_periodic_restart', fallback=False)
		self.cg_periodic_its = self.config.getint('OptimizationRoutine', 'cg_periodic_its', 10)
		self.cg_relative_restart = self.config.getboolean('OptimizationRoutine', 'cg_relative_restart', fallback=False)
		self.cg_restart_tol = self.config.getfloat('OptimizationRoutine', 'cg_restart_tol', fallback=0.25)



	def run(self):
		"""Performs the optimization via the nonlinear cg method

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
		self.memory = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False
		self.gradient.vector()[:] = 1.0

		while True:

			self.gradient_prev.vector()[:] = self.gradient.vector()[:]

			self.adjoint_problem.has_solution = False
			self.shape_gradient_problem.has_solution = False
			self.shape_gradient_problem.solve()

			self.gradient_norm = np.sqrt(self.shape_gradient_problem.gradient_norm_squared)
			if self.iteration > 0:
				if self.cg_method == 'FR':
					self.beta_numerator = self.shape_form_handler.scalar_product(self.gradient, self.gradient)
					self.beta_denominator = self.shape_form_handler.scalar_product(self.gradient_prev, self.gradient_prev)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'PR':
					self.difference.vector()[:] = self.gradient.vector()[:] - self.gradient_prev.vector()[:]

					self.beta_numerator = self.shape_form_handler.scalar_product(self.gradient, self.difference)
					self.beta_denominator = self.shape_form_handler.scalar_product(self.gradient_prev, self.gradient_prev)
					self.beta = self.beta_numerator / self.beta_denominator
					# self.beta = np.maximum(self.beta, 0.0)

				elif self.cg_method == 'HS':
					self.difference.vector()[:] = self.gradient.vector()[:] - self.gradient_prev.vector()[:]

					self.beta_numerator = self.shape_form_handler.scalar_product(self.gradient, self.difference)
					self.beta_denominator = self.shape_form_handler.scalar_product(self.difference, self.search_direction)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'DY':
					self.difference.vector()[:] = self.gradient.vector()[:] - self.gradient_prev.vector()[:]

					self.beta_numerator = self.shape_form_handler.scalar_product(self.gradient, self.gradient)
					self.beta_denominator = self.shape_form_handler.scalar_product(self.search_direction, self.difference)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'CD':
					self.beta_numerator = self.shape_form_handler.scalar_product(self.gradient, self.gradient)
					self.beta_denominator = -self.shape_form_handler.scalar_product(self.search_direction, self.gradient)
					self.beta = self.beta_numerator / self.beta_denominator

				elif self.cg_method == 'HZ':
					self.difference.vector()[:] = self.gradient.vector()[:] - self.gradient_prev.vector()[:]

					dy = self.shape_form_handler.scalar_product(self.search_direction, self.difference)
					y2 = self.shape_form_handler.scalar_product(self.difference, self.difference)

					self.difference.vector()[:] = self.difference.vector()[:] - 2*y2/dy*self.search_direction.vector()[:]

					self.beta = self.shape_form_handler.scalar_product(self.difference, self.gradient) / dy

				else:
					raise Exception('Not a valid method for nonlinear CG. Choose either FR (Fletcher Reeves), PR (Polak Ribiere), HS (Hestenes Stiefel), DY (Dai Yuan), CD (Conjugate Descent) or HZ (Hager Zhang).')

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

			self.search_direction.vector()[:] = -self.gradient.vector()[:] + self.beta*self.search_direction.vector()[:]
			if self.cg_periodic_restart:
				if self.memory < self.cg_periodic_its:
					self.memory += 1
				else:
					self.search_direction.vector()[:] = -self.gradient.vector()[:]
					self.memory = 0
			if self.cg_relative_restart:
				if abs(self.shape_form_handler.scalar_product(self.gradient, self.gradient_prev)) / pow(self.gradient_norm, 2) >= self.cg_restart_tol:
					self.search_direction.vector()[:] = -self.gradient.vector()[:]
					self.memory = 0

			self.directional_derivative = self.shape_form_handler.scalar_product(self.gradient, self.search_direction)

			if self.directional_derivative >= 0:
				self.search_direction.vector()[:] = -self.gradient.vector()[:]

			self.line_search.search(self.search_direction, self.has_curvature_info)
			if self.line_search_broken:
				if self.soft_exit:
					print('Armijo rule failed.')
					break
				else:
					raise Exception('Armijo rule failed.')

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				self.print_results()
				if self.soft_exit:
					print('Maximum number of iterations exceeded.')
					break
				else:
					raise Exception('Maximum number of iterations exceeded.')
