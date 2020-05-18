"""
Created on 24/02/2020, 14.31

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from ..line_search import ArmijoLineSearch
from ...helpers import summ



class Newton(OptimizationAlgorithm):
	
	def __init__(self, optimization_problem):
		"""A truncated Newton method (using either cg, minres or cr) to solve the optimization problem
		
		Additional parameters in the config file:
			inner_newton : (one of) cg [conjugate gradient], minres [minimal residual] or cr [conjugate residual]
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimizationProblem
			the OptimizationProblem object
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.stepsize = 1.0
		self.armijo_stepsize_initial = self.stepsize

		self.has_curvature_info = False

		self.armijo_broken = False



	def run(self):
		"""Performs the optimization via the truncated Newton method
		
		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""
		
		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		while True:
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			self.gradient_norm = np.sqrt(self.optimization_problem.stationary_measure_squared())
			if self.iteration == 0:
				self.gradient_norm_initial = self.gradient_norm
				if self.gradient_norm_initial == 0:
					self.objective_value = self.cost_functional.compute()
					self.print_results()
					break

			self.relative_norm = self.gradient_norm / self.gradient_norm_initial
			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				self.print_results()
				break

			self.search_directions = self.optimization_problem.hessian_problem.newton_solve()
			self.directional_derivative = self.form_handler.scalar_product(self.search_directions, self.gradients)
			if self.directional_derivative > 0:
				self.has_curvature_info = False
				# print('No descent direction')
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
			else:
				self.has_curvature_info = True

			self.line_search.search(self.search_directions, self.has_curvature_info)

			if self.armijo_broken and self.has_curvature_info:
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
				self.has_curvature_info = False
				self.armijo_broken = False
				self.line_search.search(self.search_directions, self.has_curvature_info)

				if self.armijo_broken:
					if self.soft_exit:
						print('Armijo rule failed.')
						break
					else:
						raise SystemExit('Armijo rule failed.')

			elif self.armijo_broken and not self.has_curvature_info:
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
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves) +
				  ' --- Sensitivity equations solved: ' + str(self.optimization_problem.hessian_problem.no_sensitivity_solves))
			print('')
