"""
Created on 06/03/2020, 10.21

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from ..line_search import ArmijoLineSearch
from ...helpers import summ



class SemiSmoothNewton(OptimizationAlgorithm):

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


		while True:
			self.state_problem.has_solution = False
			self.objective_value = self.cost_functional.compute()

			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			self.gradient_norm_squared = self.optimization_problem.stationary_measure_squared()
			if self.iteration == 0:
				self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)

			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			if self.relative_norm <= self.tolerance:
				self.converged = True
				break

			self.print_results()

			self.search_directions, self.delta_mu = self.optimization_problem.semi_smooth_hessian.newton_solve()
			self.directional_derivative = summ([fenics.assemble(fenics.inner(self.search_directions[i], self.gradients[i])*self.optimization_problem.control_measures[i]) for i in range(len(self.controls))])

			self.idx_inactive = self.optimization_problem.semi_smooth_hessian.idx_inactive
			self.idx_active = self.optimization_problem.semi_smooth_hessian.idx_active
			self.mu = self.optimization_problem.semi_smooth_hessian.mu

			for j in range(len(self.controls)):
				self.controls[j].vector()[:] += self.search_directions[j].vector()[:]
				self.optimization_problem.semi_smooth_hessian.mu[j].vector()[:] += self.delta_mu[j].vector()[:]
				self.optimization_problem.semi_smooth_hessian.mu[j].vector()[self.optimization_problem.semi_smooth_hessian.idx_inactive[j]] = 0

			if self.armijo_broken:
				print('Armijo rule failed')
				break

			self.iteration += 1

			if self.iteration >= self.maximum_iterations:
				break

		if self.converged:
			self.print_results()

		if self.verbose:
			print('')
			print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
			print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves) +
				  ' --- Sensitivity equations solved: ' + str(self.optimization_problem.semi_smooth_hessian.no_sensitivity_solves))
			print('')
