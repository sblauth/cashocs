"""
Created on 01.04.20, 14:22

@author: sebastian
"""

import fenics
import numpy as np
from ...optimization_algorithm import OptimizationAlgorithm
from .unconstrained_line_search import UnconstrainedLineSearch
from ....helpers import summ



class InnerNewton(OptimizationAlgorithm):

	def __init__(self, optimization_problem):
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = UnconstrainedLineSearch(self)
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations_inner_pdas')
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'pdas_inner_tolerance')
		self.reduced_gradient = [fenics.Function(self.optimization_problem.control_spaces[j]) for j in range(len(self.controls))]
		self.first_iteration = True
		self.first_gradient_norm = 1.0

		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.stepsize = 1.0
		self.armijo_stepsize_initial = self.stepsize

		self.armijo_broken = False

	def run(self, idx_active):

		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		while True:
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()

			for j in range(len(self.controls)):
				self.reduced_gradient[j].vector()[:] = self.gradients[j].vector()[:]
				self.reduced_gradient[j].vector()[idx_active[j]] = 0.0

			self.gradient_norm_squared = self.form_handler.scalar_product(self.reduced_gradient, self.reduced_gradient)

			if self.iteration==0:
				self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)
				if self.first_iteration:
					self.first_gradient_norm = self.gradient_norm_initial
					self.first_iteration = False

			self.relative_norm = np.sqrt(self.gradient_norm_squared)/self.gradient_norm_initial
			if self.relative_norm <= self.tolerance or self.relative_norm*self.gradient_norm_initial / self.first_gradient_norm <= self.tolerance/2:
				self.converged = True
				break

			self.search_directions = self.optimization_problem.unconstrained_hessian.newton_solve(idx_active)
			self.directional_derivative = self.form_handler.scalar_product(self.search_directions, self.reduced_gradient)
			if self.directional_derivative > 0:
				print('No descent direction')
				for i in range(len(self.controls)):
					self.search_directions[i].vector()[:] = -self.reduced_gradient[i].vector()[:]

			self.line_search.search(self.search_directions)
			if self.armijo_broken:
				raise SystemExit('Armijo rule failed.')
				# print('Armijo rule failed')
				# break

			self.iteration += 1

			if self.iteration >= self.maximum_iterations:
				raise SystemExit('Maximum number of iterations exceeded.')

		# if self.converged:
		# 	self.print_results()
