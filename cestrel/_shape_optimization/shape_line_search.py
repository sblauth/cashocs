"""
Created on 15/06/2020, 08.00

@author: blauths
"""

import fenics
import numpy as np



class ArmijoLineSearch:

	def __init__(self, optimization_algorithm):
		"""Initializes the line search

		Parameters
		----------
		optimization_algorithm : cestrel.shape_optimization.shape_optimization_algorithm.ShapeOptimizationAlgorithm
			the optimization problem of interest
		"""

		self.optimization_algorithm = optimization_algorithm
		self.config = self.optimization_algorithm.config
		self.optimization_problem = self.optimization_algorithm.optimization_problem
		self.shape_form_handler = self.optimization_problem.shape_form_handler
		self.mesh_handler = self.optimization_problem.mesh_handler
		# self.mesh_handler = _MeshHandler(self.shape_form_handler)
		self.deformation = fenics.Function(self.shape_form_handler.deformation_space)

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial', fallback=1.0)
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo', fallback=1e-4)
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo', fallback=2.0)
		self.armijo_stepsize_initial = self.stepsize

		self.cost_functional = self.optimization_problem.reduced_cost_functional

		self.gradient = self.optimization_algorithm.gradient

		algorithm = self.config.get('OptimizationRoutine', 'algorithm')
		self.is_newton_like = algorithm in ['lbfgs']
		self.is_newton = algorithm in ['newton', 'semi_smooth_newton']
		self.is_steepest_descent = algorithm in ['gradient_descent']
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

		return self.stepsize*self.shape_form_handler.scalar_product(self.gradient, search_direction)



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
			if self.stepsize*self.search_direction_inf <= 1e-8:
				print('\nStepsize too small.')
				self.optimization_algorithm.line_search_broken = True
				break
			elif not self.is_newton_like and not self.is_newton and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
				print('\nStepsize too small.')
				self.optimization_algorithm.line_search_broken = True
				break

			self.deformation.vector()[:] = self.stepsize*search_direction.vector()[:]

			if self.mesh_handler.move_mesh(self.deformation):
				self.mesh_handler.compute_relative_quality()

				if self.mesh_handler.min_quality < self.mesh_handler.mesh_quality_tol / 10:
					self.stepsize /= self.beta_armijo
					self.mesh_handler.revert_transformation()
					continue

				self.optimization_algorithm.state_problem.has_solution = False
				self.objective_step = self.cost_functional.evaluate()

				if self.objective_step < self.optimization_algorithm.objective_value + self.epsilon_armijo*self.decrease_measure(search_direction):
					if self.mesh_handler.min_quality < self.mesh_handler.mesh_quality_tol:
						self.stepsize /= self.beta_armijo
						self.mesh_handler.revert_transformation()
						self.optimization_algorithm.line_search_broken = True
						print('\nMesh Quality too low.')
						self.mesh_handler.remesh()

					if self.optimization_algorithm.iteration == 0:
						self.armijo_stepsize_initial = self.stepsize
					self.shape_form_handler.update_scalar_product()
					break

				else:
					self.stepsize /= self.beta_armijo
					self.mesh_handler.revert_transformation()

			else:
				self.stepsize /= self.beta_armijo

		if not self.optimization_algorithm.line_search_broken:
			self.optimization_algorithm.stepsize = self.stepsize
			self.optimization_algorithm.objective_value = self.objective_step

		if not has_curvature_info:
			self.stepsize *= self.beta_armijo
