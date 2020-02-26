"""
Created on 24/02/2020, 15.40

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm



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
		
		self.gradient_problem = self.optimization_problem.gradient_problem
		
		self.gradient = self.optimization_problem.gradient
		self.control = self.optimization_problem.control
		
		self.control_temp = fenics.Function(self.optimization_problem.control_space)
		self.search_direction = fenics.Function(self.optimization_problem.control_space)
		self.gradient_prev = fenics.Function(self.optimization_problem.control_space)
		
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		
		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial')
		self.armijo_stepsize_initial = self.stepsize
		
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'tolerance')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations')
		self.cg_method = self.config.get('OptimizationRoutine', 'cg_method')
		
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
	
	
	
	def run(self):
		"""Performs the optimization via the nonlinear cg method
		
		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""
		
		self.iteration = 0
		self.objective_value = self.cost_functional.compute()
		self.restart = 0
		
		self.gradient_problem.has_solution = False
		self.gradient_problem.solve()
		self.search_direction.vector()[:] = -self.gradient.vector()[:]
		self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
		self.gradient_norm_initial = np.sqrt(self.gradient_norm_squared)
		
		self.gradient_norm_inf = np.max(np.abs(self.gradient.vector()[:]))
		self.relative_norm = 1.0
		
		self.print_results()
		
		while self.relative_norm > self.tolerance:
			self.gradient_prev.vector()[:] = self.gradient.vector()[:]
			
			self.directional_derivative = fenics.assemble(fenics.inner(self.gradient, self.search_direction)*self.optimization_problem.control_measure)
			if self.directional_derivative >= 0:
				self.search_direction.vector()[:] = -self.gradient.vector()[:]
			
			self.control_temp.vector()[:] = self.control.vector()[:]
			
			# Armijo Line Search
			while True:
				if self.stepsize*self.gradient_norm_inf <= 1e-10:
					self.armijo_broken = True
					break
				elif self.stepsize/self.armijo_stepsize_initial <= 1e-8:
					self.armijo_broken = True
					break
				
				self.control.vector()[:] += self.stepsize*self.search_direction.vector()[:]
				
				self.state_problem.has_solution = False
				self.objective_step = self.cost_functional.compute()
				
				if self.objective_step < self.objective_value + self.epsilon_armijo*self.stepsize*self.directional_derivative:
					if self.iteration == 0:
						self.armijo_stepsize_initial = self.stepsize
					break
					
				else:
					self.stepsize /= self.beta_armijo
					self.control.vector()[:] = self.control_temp.vector()[:]
				
			
			if self.armijo_broken:
				print('Armijo rule failed')
				break
			
			self.objective_value = self.objective_step
			
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			
			self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			self.gradient_norm_inf = np.max(np.abs(self.gradient.vector()[:]))
			
			if self.cg_method == 'FR':
				self.beta_numerator = fenics.assemble(fenics.inner(self.gradient, self.gradient)*self.optimization_problem.control_measure)
				self.beta_denominator = fenics.assemble(fenics.inner(self.gradient_prev, self.gradient_prev)*self.optimization_problem.control_measure)
				self.beta = self.beta_numerator / self.beta_denominator
				
			elif self.cg_method == 'PR':
				self.beta_numerator = fenics.assemble(fenics.inner(self.gradient, self.gradient - self.gradient_prev)*self.optimization_problem.control_measure)
				self.beta_denominator = fenics.assemble(fenics.inner(self.gradient_prev, self.gradient_prev)*self.optimization_problem.control_measure)
				self.beta = self.beta_numerator / self.beta_denominator
				self.beta = np.maximum(self.beta, 0)
				
			elif self.cg_method == 'HS':
				self.beta_numerator = fenics.assemble(fenics.inner(self.gradient, self.gradient - self.gradient_prev)*self.optimization_problem.control_measure)
				self.beta_denominator = fenics.assemble(fenics.inner(self.gradient - self.gradient_prev, self.search_direction)*self.optimization_problem.control_measure)
				self.beta = self.beta_numerator / self.beta_denominator
				
			elif self.cg_method == 'DY':
				self.beta_numerator = fenics.assemble(fenics.inner(self.gradient, self.gradient)*self.optimization_problem.control_measure)
				self.beta_denominator = fenics.assemble(fenics.inner(self.search_direction, self.gradient - self.gradient_prev)*self.optimization_problem.control_measure)
				self.beta = self.beta_numerator / self.beta_denominator
			
			elif self.cg_method == 'CD':
				self.beta_numerator = fenics.assemble(fenics.inner(self.gradient, self.gradient)*self.optimization_problem.control_measure)
				self.beta_denominator = -fenics.assemble(fenics.inner(self.search_direction, self.gradient)*self.optimization_problem.control_measure)
				self.beta = self.beta_numerator / self.beta_denominator
				
			elif self.cg_method == 'HZ':
				dy = fenics.assemble(fenics.inner(self.search_direction, self.gradient - self.gradient_prev)*self.optimization_problem.control_measure)
				y2 = fenics.assemble(fenics.inner(self.gradient - self.gradient_prev, self.gradient - self.gradient_prev)*self.optimization_problem.control_measure)
				self.beta = fenics.assemble(fenics.inner(self.gradient - self.gradient_prev - 2*self.search_direction*fenics.Constant(y2/dy),
														 self.gradient/fenics.Constant(dy))*self.optimization_problem.control_measure)
			
			else:
				raise SystemExit('Not a valid method for nonlinear CG. Choose either FR (Fletcher Reeves), PR (Polak Ribiere), HS (Hestenes Stiefel), DY (Dai Yuan), CD (Conjugate Descent) or HZ (Hager Zhang).')
			
			if self.restart < 2:
				self.search_direction.vector()[:] = -self.gradient.vector()[:] + self.beta*self.search_direction.vector()[:]
				# self.restart += 1
			else:
				self.search_direction.vector()[:] = -self.gradient.vector()[:]
				self.restart = 0
			
			self.iteration += 1
			self.print_results()
			
			if self.iteration >= self.maximum_iterations:
				break
			
			self.stepsize *= self.beta_armijo
			
		print('')
		print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
			  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
		print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
			  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
		print('')
