"""
Created on 24/02/2020, 14.31

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm



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
		
		self.gradient = self.optimization_problem.gradient
		self.control = self.optimization_problem.control
		
		self.control_temp = fenics.Function(self.optimization_problem.control_space)
		
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'tolerance')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations')
		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial')
		self.armijo_stepsize_initial = self.stepsize
		
		
	
	def print_results(self):
		"""Prints the current state of the optimization algorithm to the console.
		
		Returns
		-------
		None
			see method description

		"""
		if self.iteration == 0:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.gradient_norm_initial, '.3e') + ' (abs)' + ' \n '
		else:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel) '
		
		if self.verbose:
			print(output)



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
		self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
		self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)
		
		self.gradient_norm_inf = np.max(np.abs(self.gradient.vector()[:]))
		self.relative_norm = 1.0
		
		self.print_results()
		
		while self.relative_norm > self.tolerance:
			
			self.control_temp.vector()[:] = self.control.vector()[:]
			
			self.delta_control = self.optimization_problem.hessian_problem.newton_solve()
			self.directional_derivative = fenics.assemble(fenics.inner(self.delta_control, self.gradient)*self.optimization_problem.control_measure)
			
			### TODO: Implement a damping based on either residual or increment (and also using the gradient direction if Newton-Direction does not give descent
			if self.directional_derivative > 0:
				self.delta_control.vector()[:] = -self.delta_control.vector()[:]
			
			self.control.vector()[:] += self.delta_control.vector()[:]

			self.state_problem.has_solution = False
			self.objective_value = self.cost_functional.compute()
			
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			
			self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			self.gradient_norm_inf = np.max(np.abs(self.gradient.vector()[:]))
			
			self.iteration += 1
			self.print_results()
			
			if self.iteration >= self.maximum_iterations:
				break
				
		print('')
		print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
			  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
		print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
			  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
		print('')
