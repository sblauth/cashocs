"""
Created on 24/02/2020, 14.31

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
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
		self.gradient_problem = self.optimization_problem.gradient_problem
		
		self.gradients = self.optimization_problem.gradients
		self.controls = self.optimization_problem.controls

		self.control_constraints = self.optimization_problem.control_constraints
		
		self.controls_temp = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.projected_difference = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'tolerance')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations')
		self.stepsize = 1.0
		self.armijo_stepsize_initial = self.stepsize

		self.armijo_broken = False
		
		
	
	def print_results(self):
		"""Prints the current state of the optimization algorithm to the console.
		
		Returns
		-------
		None
			see method description

		"""

		if self.iteration == 0:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.gradient_norm_initial, '.3e') + ' (abs)    Step size:  ' + format(self.stepsize, '.3e') + ' \n '
		else:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)    Step size:  ' + format(self.stepsize, '.3e')
		
		if self.verbose:
			print(output)



	def project(self, a):

		self.control_constraints = self.optimization_problem.control_constraints

		for j in range(self.form_handler.control_dim):
			a[j].vector()[:] = np.maximum(self.control_constraints[j][0], np.minimum(self.control_constraints[j][1], a[j].vector()[:]))

		return a



	def stationary_measure_squared(self):

		for j in range(self.form_handler.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.gradients[j].vector()[:]

		self.project(self.projected_difference)

		for j in range(self.form_handler.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.projected_difference[j].vector()[:]

		return self.scalar_product(self.projected_difference, self.projected_difference)



	def scalar_product(self, a, b):
		"""Implements the scalar product needed for the algorithm

		Parameters
		----------
		a : List[dolfin.function.function.Function]
			The first input
		b : List[dolfin.function.function.Function]
			The second input

		Returns
		-------
		 : float
			The value of the scalar product

		"""

		return summ([fenics.assemble(fenics.inner(a[i], b[i])*self.optimization_problem.control_measures[i]) for i in range(len(self.gradients))])



	def run(self):
		"""Performs the optimization via the truncated Newton method
		
		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""
		
		self.iteration = 0
		self.objective_value = self.cost_functional.compute()
		
		self.gradient_problem.has_solution = False
		self.gradient_problem.solve()
		self.gradient_norm_squared = self.stationary_measure_squared()
		# self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
		self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)
		
		self.gradient_norm_inf = np.max([np.max(np.abs(self.gradients[i].vector()[:])) for i in range(len(self.controls))])
		self.relative_norm = 1.0
		
		self.print_results()
		
		while self.relative_norm > self.tolerance:
			self.stepsize = 1.0
			for i in range(len(self.controls)):
				self.controls_temp[i].vector()[:] = self.controls[i].vector()[:]
			
			self.delta_control = self.optimization_problem.hessian_problem.newton_solve()
			self.directional_derivative = summ([fenics.assemble(fenics.inner(self.delta_control[i], self.gradients[i])*self.optimization_problem.control_measures[i]) for i in range(len(self.controls))])
			
			if self.directional_derivative > 0:
				print('No descent direction')
				for i in range(len(self.gradients)):
					# self.delta_control[i].vector()[:] = -self.delta_control[i].vector()[:]
					self.delta_control[i].vector()[:] = -self.gradients[i].vector()[:]

			self.search_direction_inf = np.max([np.max(np.abs(self.delta_control[i].vector()[:])) for i in range(len(self.gradients))])

			# Armijo Line Search
			while True:
				if self.stepsize*self.search_direction_inf <= 1e-10:
					self.armijo_broken = True
					break
				elif self.iteration > 0 and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
					self.armijo_broken = True
					break

				for i in range(len(self.controls)):
					self.controls[i].vector()[:] += self.stepsize*self.delta_control[i].vector()[:]

				self.project(self.controls)

				self.state_problem.has_solution = False
				self.objective_step = self.cost_functional.compute()

				for j in range(self.form_handler.control_dim):
					self.projected_difference[j].vector()[:] = self.controls_temp[j].vector()[:] - self.controls[j].vector()[:]

				if self.objective_step < self.objective_value - self.epsilon_armijo*self.scalar_product(self.gradients, self.projected_difference):
					if self.iteration == 0:
						self.armijo_stepsize_initial = self.stepsize
					break

				else:
					self.stepsize /= self.beta_armijo
					for i in range(len(self.controls)):
						self.controls[i].vector()[:] = self.controls_temp[i].vector()[:]


			if self.armijo_broken:
				print('Armijo rule failed')
				break

			self.objective_value = self.objective_step

			
			# for i in range(len(self.controls)):
			# 	self.controls[i].vector()[:] += self.delta_control[i].vector()[:]

			# self.state_problem.has_solution = False
			# self.objective_value = self.cost_functional.compute()
			
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			
			self.gradient_norm_squared = self.stationary_measure_squared()
			# self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			self.gradient_norm_inf = np.max([np.max(np.abs(self.gradients[i].vector()[:])) for i in range(len(self.gradients))])
			
			self.iteration += 1
			self.print_results()
			
			if self.iteration >= self.maximum_iterations:
				break
				
		print('')
		print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
			  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
		print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
			  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves) +
			  ' --- Sensitivity equations solved: ' + str(self.optimization_problem.hessian_problem.no_sensitivity_solves))
		print('')
