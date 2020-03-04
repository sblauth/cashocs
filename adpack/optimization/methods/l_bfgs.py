"""
Created on 24/02/2020, 13.11

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from ...helpers import summ
from _collections import deque



class LBFGS(OptimizationAlgorithm):
	
	def __init__(self, optimization_problem):
		"""Implements the L-BFGS method for solving the optimization problem
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimizationProblem
			instance of the OptimizationProblem (user defined)
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)
		
		self.gradient_problem = self.optimization_problem.gradient_problem
		
		self.gradients = self.optimization_problem.gradients
		self.controls = self.optimization_problem.controls
		
		self.controls_temp = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		
		self.stepsize = self.config.getfloat('OptimizationRoutine', 'step_initial')
		self.armijo_stepsize_initial = self.stepsize
		
		self.q = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.temp = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.storage_y = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.storage_s = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.projected_difference = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.tolerance = self.config.getfloat('OptimizationRoutine', 'tolerance')
		self.epsilon_armijo = self.config.getfloat('OptimizationRoutine', 'epsilon_armijo')
		self.beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations')
		self.memory_vectors = self.config.getint('OptimizationRoutine', 'memory_vectors')
		self.use_bfgs_scaling = self.config.getboolean('OptimizationRoutine', 'use_bfgs_scaling')

		self.control_constraints = self.optimization_problem.control_constraints
		
		if self.memory_vectors > 0:
			self.history_s = deque()
			self.history_y = deque()
			self.history_rho = deque()
			self.gradients_prev = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
			self.y_k = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
			self.s_k = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		
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
	
	
	
	def bfgs_scalar_product(self, a, b):
		"""A short cut for computing the scalar product in the BFGS double loop
		
		Parameters
		----------
		a : List[dolfin.function.function.Function]
			first input
		b : List[dolfin.function.function.Function]
			second input
			
		Returns
		-------
		 : float
			the value of the scalar product

		"""
		
		return summ([fenics.assemble(fenics.inner(a[i], b[i])*self.optimization_problem.control_measures[i]) for i in range(len(self.controls))])



	def project_active(self, a, b):

		for j in range(self.form_handler.control_dim):
			self.temp[j].vector()[:] = 0.0
			idx = np.asarray(np.logical_or(self.controls[j].vector()[:] <= self.control_constraints[j][0], self.controls[j].vector()[:] >= self.control_constraints[j][1])).nonzero()[0]
			self.temp[j].vector()[idx] = a[j].vector()[idx]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_inactive(self, a, b):

		for j in range(self.form_handler.control_dim):
			self.temp[j].vector()[:] = 0.0
			idx = np.asarray(np.invert(np.logical_or(self.controls[j].vector()[:] <= self.control_constraints[j][0], self.controls[j].vector()[:] >= self.control_constraints[j][1]))).nonzero()[0]
			self.temp[j].vector()[idx] = a[j].vector()[idx]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project(self, a):

		self.control_constraints = self.optimization_problem.control_constraints

		for j in range(self.form_handler.control_dim):
			a[j].vector()[:] = np.maximum(self.control_constraints[j][0], np.minimum(self.control_constraints[j][1], a[j].vector()[:]))

		return a


	
	def compute_search_direction(self, grad):
		"""Computes the search direction for the BFGS method with the so-called double loop
		
		Parameters
		----------
		grad : List[dolfin.function.function.Function]
			the current gradient

		Returns
		-------
		self.q : dolfin.function.function.Function
			a function corresponding to the current / next search direction

		"""
		
		if self.memory_vectors > 0 and len(self.history_s) > 0:
			history_alpha = deque()
			for j in range(len(self.controls)):
				self.q[j].vector()[:] = grad[j].vector()[:]

			self.project_inactive(self.q, self.q)
				
			for i, _ in enumerate(self.history_s):
				alpha = self.history_rho[i]*self.bfgs_scalar_product(self.history_s[i], self.q)
				history_alpha.append(alpha)
				for j in range(len(self.controls)):
					self.q[j].vector()[:] -= alpha*self.history_y[i][j].vector()[:]
			
			if self.use_bfgs_scaling and self.iteration > 0:
				factor = self.bfgs_scalar_product(self.history_y[0], self.history_s[0])/self.bfgs_scalar_product(self.history_y[0], self.history_y[0])
			else:
				factor = 1.0
			
			for j in range(len(self.controls)):
				self.q[j].vector()[:] *= factor

			self.project_inactive(self.q, self.q)
			
			for i, _ in enumerate(self.history_s):
				beta = self.history_rho[-1 - i]*self.bfgs_scalar_product(self.history_y[-1 - i], self.q)
				
				for j in range(len(self.controls)):
					self.q[j].vector()[:] += self.history_s[-1 - i][j].vector()[:]*(history_alpha[-1 - i] - beta)

			self.project_inactive(self.q, self.q)
			self.project_active(self.gradients, self.temp)
			for j in range(len(self.controls)):
				self.q[j].vector()[:] += self.temp[j].vector()[:]
				self.q[j].vector()[:] *= -1
		
		else:
			for j in range(len(self.controls)):
				self.q[j].vector()[:] = - grad[j].vector()[:]
		
		return self.q



	def stationary_measure_squared(self):

		for j in range(self.form_handler.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.gradients[j].vector()[:]

		self.project(self.projected_difference)

		for j in range(self.form_handler.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.projected_difference[j].vector()[:]

		return self.bfgs_scalar_product(self.projected_difference, self.projected_difference)


	
	def run(self):
		"""Performs the optimization via the limited memory BFGS method
		
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
		self.gradient_norm_inf = np.max([np.max(np.abs(self.gradients[i].vector()[:])) for i in range(len(self.gradients))])
		
		self.relative_norm = 1.0
		
		self.print_results()
		
		while self.relative_norm > self.tolerance:
			for i in range(len(self.controls)):
				self.controls_temp[i].vector()[:] = self.controls[i].vector()[:]
			self.search_directions = self.compute_search_direction(self.gradients)
			self.search_direction_inf = np.max([np.max(np.abs(self.search_directions[i].vector()[:])) for i in range(len(self.gradients))])
			
			self.directional_derivative = self.bfgs_scalar_product(self.search_directions, self.gradients)

			# Armijo Line Search
			while True:
				if self.stepsize*self.search_direction_inf <= 1e-10:
					self.armijo_broken = True
					break
				elif self.memory_vectors == 0 and self.iteration > 0 and self.stepsize/self.armijo_stepsize_initial <= 1e-8:
					self.armijo_broken = True
					break
				
				for i in range(len(self.controls)):
					self.controls[i].vector()[:] += self.stepsize*self.search_directions[i].vector()[:]

				self.project(self.controls)

				self.state_problem.has_solution = False
				self.objective_step = self.cost_functional.compute()

				for j in range(self.form_handler.control_dim):
					self.projected_difference[j].vector()[:] = self.controls_temp[j].vector()[:] - self.controls[j].vector()[:]

				if self.objective_step < self.objective_value - self.epsilon_armijo*self.bfgs_scalar_product(self.gradients, self.projected_difference):
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
			
			if self.memory_vectors > 0:
				for i in range(len(self.gradients)):
					self.gradients_prev[i].vector()[:] = self.gradients[i].vector()[:]
			
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			
			self.gradient_norm_squared = self.stationary_measure_squared()
			# self.gradient_norm_squared = self.gradient_problem.return_norm_squared()
			self.relative_norm = np.sqrt(self.gradient_norm_squared) / self.gradient_norm_initial
			self.gradient_norm_inf = np.max([np.max(np.abs(self.gradients[i].vector()[:])) for i in range(len(self.gradients))])
			
			if self.memory_vectors > 0:
				for i in range(len(self.gradients)):
					self.storage_y[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]
					self.storage_s[i].vector()[:] = self.stepsize*self.search_directions[i].vector()[:]

				self.project_inactive(self.storage_y, self.y_k)
				self.project_inactive(self.storage_s, self.s_k)

				self.history_y.appendleft([x.copy(True) for x in self.y_k])
				self.history_s.appendleft([x.copy(True) for x in self.s_k])
				rho = 1/self.bfgs_scalar_product(self.y_k, self.s_k)
				self.history_rho.appendleft(rho)
				
				if 1/rho <= 0:
					self.history_s = deque()
					self.history_y = deque()
					self.history_rho = deque()
				
				if len(self.history_s) > self.memory_vectors:
					self.history_s.pop()
					self.history_y.pop()
					self.history_rho.pop()
			
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
