"""
Created on 15/06/2020, 08.01

@author: blauths
"""

import fenics
import numpy as np
from ..shape_optimization_algorithm import OptimizationAlgorithm
from ..shape_line_search import ArmijoLineSearch



class CG(OptimizationAlgorithm):
	def __init__(self, optimization_problem):
		"""A nonlinear cg method to solve the optimization problem

		Additional parameters in the config file:
			cg_method : (one of) FR [Fletcher Reeves], PR [Polak Ribiere], HS [Hestenes Stiefel], DY [Dai-Yuan], CD [Conjugate Descent], HZ [Hager Zhang]

		Parameters
		----------
		optimization_problem : adpack.shape_optimization.shape_optimization_problem.ShapeOptimizationProblem
			the OptimalControlProblem object
		"""

		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.gradient_prev = fenics.Function(self.shape_form_handler.deformation_space)
		self.difference = fenics.Function(self.shape_form_handler.deformation_space)
		self.temp_HZ = fenics.Function(self.shape_form_handler.deformation_space)

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.armijo_stepsize_initial = self.stepsize
		self.armijo_broken = False

		self.cg_method = self.config.get('OptimizationRoutine', 'cg_method')
		self.cg_periodic_restart = self.config.getboolean('OptimizationRoutine', 'cg_periodic_restart')
		self.cg_periodic_its = self.config.getint('OptimizationRoutine', 'cg_periodic_its')
		self.cg_relative_restart = self.config.getboolean('OptimizationRoutine', 'cg_relative_restart')
		self.cg_restart_tol = self.config.getfloat('OptimizationRoutine', 'cg_restart_tol')

		self.has_curvature_info = False



	def run(self):
		"""Performs the optimization via the nonlinear cg method

		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""

		self.iteration = self.config.getint('OptimizationRoutine', 'iteration_counter', fallback=0)
		self.gradient_norm_initial = self.config.getfloat('OptimizationRoutine', 'gradient_norm_initial', fallback=0.0)
		self.memory = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False
		self.gradient.vector()[:] = 1.0

		while True:

			self.gradient_prev.vector()[:] = self.gradient.vector()[:]

			self.adjoint_problem.has_solution = False
			self.shape_gradient_problem.has_solution = False
			self.shape_gradient_problem.solve()

			self.gradient_norm = np.sqrt(self.optimization_problem.norm_squared())
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
					raise SystemExit('Not a valid method for nonlinear CG. Choose either FR (Fletcher Reeves), PR (Polak Ribiere), HS (Hestenes Stiefel), DY (Dai Yuan), CD (Conjugate Descent) or HZ (Hager Zhang).')

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
					raise SystemExit('Armijo rule failed.')

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				self.print_results()
				if self.soft_exit:
					print('Maximum number of iterations exceeded.')
					break
				else:
					raise SystemExit('Maximum number of iterations exceeded.')

		if self.verbose:
			print('')
			print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
			print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
			print('')
