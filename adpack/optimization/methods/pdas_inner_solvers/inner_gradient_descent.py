"""
Created on 01.04.20, 11:43

@author: sebastian
"""

import numpy as np
from ...optimization_algorithm import OptimizationAlgorithm
from .unconstrained_line_search import UnconstrainedLineSearch
import fenics
import matplotlib.pyplot as plt


class InnerGradientDescent(OptimizationAlgorithm):

	def __init__(self, optimization_problem):
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = UnconstrainedLineSearch(self)
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations_inner_pdas')
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'pdas_inner_tolerance')
		self.reduced_gradient = [fenics.Function(self.optimization_problem.control_spaces[j]) for j in range(len(self.controls))]
		self.first_iteration = True
		self.first_gradient_norm = 1.0


	def run(self, idx_active):
		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		self.converged = False
		self.line_search.stepsize = 1.0

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
			if self.relative_norm <= self.tolerance or self.relative_norm*self.gradient_norm_initial/self.first_gradient_norm <= self.tolerance/2:
				self.converged = True
				# self.print_results()
				break

			for i in range(len(self.controls)):
				self.controls_temp[i].vector()[:] = self.controls[i].vector()[:]
				self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
				self.search_directions[i].vector()[idx_active[i]] = 0

			self.line_search.search(self.search_directions)
			if self.line_search_broken:
				# print('Armijo rule failed')
				for j in range(len(self.controls)):
					self.controls[j].vector()[:] = self.controls_temp[j].vector()[:]
				raise SystemExit('Armijo rule failed.')
				# break

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				print('Inner Solver exceeded maximum iterations')
				break



