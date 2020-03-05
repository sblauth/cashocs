"""
Created on 24/02/2020, 15.40

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from ..line_search import ArmijoLineSearch
from ...helpers import summ



class CG(OptimizationAlgorithm):
	
	def __init__(self, optimization_problem):
		"""A nonlinear cg method to solve the optimization problem
		
		Additional parameters in the config file:
			cg_method : (one of) FR [Fletcher Reeves], PR [Polak Ribiere], HS [Hestenes Stiefel], DY [Dai-Yuan], CD [Conjugate Descent], HZ [Hager Zhang]
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimizationProblem
			the OptimizationProblem object
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)

		self.gradients_prev = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.differences = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.temp_HZ = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.armijo_stepsize_initial = self.stepsize
		self.armijo_broken = False

		self.cg_method = self.config.get('OptimizationRoutine', 'cg_method')
		self.cg_use_restart = self.config.getboolean('OptimizationRoutine', 'cg_use_restart')
		self.cg_restart_its = self.config.getint('OptimizationRoutine', 'cg_restart_its')



	def project_direction(self, a):
		self.control_constraints = self.optimization_problem.control_constraints

		for j in range(self.form_handler.control_dim):
			idx = np.asarray(np.invert(np.logical_or(self.controls[j].vector()[:] <= self.control_constraints[j][0], self.controls[j].vector()[:] >= self.control_constraints[j][1]))).nonzero()[0]
			a[j].vector()[idx] = -self.gradients[j].vector()[idx]

	
	
	def run(self):
		"""Performs the optimization via the nonlinear cg method
		
		Returns
		-------
		None
			the result can be found in the control (user defined)

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

			self.gradient_norm_squared = self.optimization_problem.stationary_measure_squared()

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
				self.beta_denominator = self.form_handler.scalar_product(self.search_directions, self.gradients)
				self.beta = self.beta_numerator / self.beta_denominator

			elif self.cg_method == 'HZ':
				for i in range(len(self.gradients)):
					self.differences[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]

				dy = self.form_handler.scalar_product(self.search_directions, self.differences)
				y2 = self.form_handler.scalar_product(self.differences, self.differences)

				for i in range(len(self.gradients)):
					self.differences[i].vector()[:] = self.differences[i].vector()[:] - 2*y2/dy*self.search_directions[i].vector()[:]

				self.beta = self.form_handler.scalar_product(self.differences, self.gradients)/dy

			else:
				raise SystemExit('Not a valid method for nonlinear CG. Choose either FR (Fletcher Reeves), PR (Polak Ribiere), HS (Hestenes Stiefel), DY (Dai Yuan), CD (Conjugate Descent) or HZ (Hager Zhang).')

			if self.iteration == 0:
				self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)
				self.beta = 0.0

			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			if self.relative_norm <= self.tolerance:
				self.converged = True
				break

			if not self.cg_use_restart:
				for i in range(self.form_handler.control_dim):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:] + self.beta*self.search_directions[i].vector()[:]
			elif self.memory < self.cg_restart_its:
				for i in range(self.form_handler.control_dim):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:] + self.beta*self.search_directions[i].vector()[:]
				self.memory += 1
			else:
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]
				self.memory = 0

			self.project_direction(self.search_directions)
			self.directional_derivative = self.form_handler.scalar_product(self.gradients, self.search_directions)
			
			if self.directional_derivative >= 0:
				for i in range(len(self.gradients)):
					self.search_directions[i].vector()[:] = -self.gradients[i].vector()[:]

			self.line_search.search(self.search_directions)
			if self.armijo_broken:
				print('Armijo rule failed')
				break

			self.iteration += 1
			if self.iteration >= self.maximum_iterations:
				break

		if self.converged:
			self.print_results()

		print('')
		print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
			  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
		print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
			  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
		print('')
