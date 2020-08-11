"""
Created on 24/02/2020, 13.11

@author: blauths
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from ..line_search import ArmijoLineSearch
from _collections import deque



class LBFGS(OptimizationAlgorithm):
	
	def __init__(self, optimization_problem):
		"""Implements the L-BFGS method for solving the optimization problem
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimalControlProblem
			instance of the OptimalControlProblem (user defined)
		"""
		
		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.line_search = ArmijoLineSearch(self)
		self.converged = False

		self.temp = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.storage_y = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.storage_s = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.memory_vectors = self.config.getint('OptimizationRoutine', 'memory_vectors')
		self.use_bfgs_scaling = self.config.getboolean('OptimizationRoutine', 'use_bfgs_scaling')
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')

		self.has_curvature_info = False

		if self.memory_vectors > 0:
			self.history_s = deque()
			self.history_y = deque()
			self.history_rho = deque()
			self.gradients_prev = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
			self.y_k = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
			self.s_k = [fenics.Function(V) for V in self.optimization_problem.control_spaces]


	
	def compute_search_direction(self, grad):
		"""Computes the search direction for the BFGS method with the so-called double loop
		
		Parameters
		----------
		grad : list[dolfin.function.function.Function]
			the current gradient

		Returns
		-------
		self.search_directions : list[dolfin.function.function.Function]
			a function corresponding to the current / next search direction

		"""
		
		if self.memory_vectors > 0 and len(self.history_s) > 0:
			history_alpha = deque()
			for j in range(len(self.controls)):
				self.search_directions[j].vector()[:] = grad[j].vector()[:]

			self.form_handler.project_inactive(self.search_directions, self.search_directions)
				
			for i, _ in enumerate(self.history_s):
				alpha = self.history_rho[i]*self.form_handler.scalar_product(self.history_s[i], self.search_directions)
				history_alpha.append(alpha)
				for j in range(len(self.controls)):
					self.search_directions[j].vector()[:] -= alpha * self.history_y[i][j].vector()[:]
			
			if self.use_bfgs_scaling and self.iteration > 0:
				factor = self.form_handler.scalar_product(self.history_y[0], self.history_s[0]) / self.form_handler.scalar_product(self.history_y[0], self.history_y[0])
			else:
				factor = 1.0
			
			for j in range(len(self.controls)):
				self.search_directions[j].vector()[:] *= factor

			self.form_handler.project_inactive(self.search_directions, self.search_directions)
			
			for i, _ in enumerate(self.history_s):
				beta = self.history_rho[-1 - i]*self.form_handler.scalar_product(self.history_y[-1 - i], self.search_directions)
				
				for j in range(len(self.controls)):
					self.search_directions[j].vector()[:] += self.history_s[-1 - i][j].vector()[:] * (history_alpha[-1 - i] - beta)

			self.form_handler.project_inactive(self.search_directions, self.search_directions)
			self.form_handler.project_active(self.gradients, self.temp)
			for j in range(len(self.controls)):
				self.search_directions[j].vector()[:] += self.temp[j].vector()[:]
				self.search_directions[j].vector()[:] *= -1
		
		else:
			for j in range(len(self.controls)):
				self.search_directions[j].vector()[:] = - grad[j].vector()[:]
		
		return self.search_directions


	
	def run(self):
		"""Performs the optimization via the limited memory BFGS method
		
		Returns
		-------
		None
			the result can be found in the control (user defined)

		"""
		
		self.iteration = 0
		self.relative_norm = 1.0
		self.state_problem.has_solution = False

		self.adjoint_problem.has_solution = False
		self.gradient_problem.has_solution = False
		self.gradient_problem.solve()
		self.gradient_norm = np.sqrt(self.optimization_problem.stationary_measure_squared())
		self.gradient_norm_initial = self.gradient_norm
		if self.gradient_norm_initial == 0:
			self.converged = True
			self.print_results()
		self.form_handler.compute_active_sets()

		while not self.converged:
			self.search_directions = self.compute_search_direction(self.gradients)

			self.directional_derivative = self.form_handler.scalar_product(self.search_directions, self.gradients)
			if self.directional_derivative > 0:
				# print('No descent direction found')
				for j in range(self.form_handler.control_dim):
					self.search_directions[j].vector()[:] = -self.gradients[j].vector()[:]

			self.line_search.search(self.search_directions, self.has_curvature_info)
			if self.line_search_broken:
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

			if self.memory_vectors > 0:
				for i in range(len(self.gradients)):
					self.gradients_prev[i].vector()[:] = self.gradients[i].vector()[:]
			
			self.adjoint_problem.has_solution = False
			self.gradient_problem.has_solution = False
			self.gradient_problem.solve()
			self.form_handler.compute_active_sets()
			
			self.gradient_norm = np.sqrt(self.optimization_problem.stationary_measure_squared())
			self.relative_norm = self.gradient_norm / self.gradient_norm_initial

			if self.gradient_norm <= self.atol + self.rtol*self.gradient_norm_initial:
				self.print_results()
				break

			if self.memory_vectors > 0:
				for i in range(len(self.gradients)):
					self.storage_y[i].vector()[:] = self.gradients[i].vector()[:] - self.gradients_prev[i].vector()[:]
					self.storage_s[i].vector()[:] = self.stepsize*self.search_directions[i].vector()[:]

				self.form_handler.project_inactive(self.storage_y, self.y_k)
				self.form_handler.project_inactive(self.storage_s, self.s_k)

				self.history_y.appendleft([x.copy(True) for x in self.y_k])
				self.history_s.appendleft([x.copy(True) for x in self.s_k])
				self.curvature_condition = self.form_handler.scalar_product(self.y_k, self.s_k)


				if self.curvature_condition <= 1e-14:
				# if self.curvature_condition <= 0.0:
				# if self.curvature_condition / self.form_handler.scalar_product(self.s_k, self.s_k) < 1e-7 * self.gradient_problem.return_norm_squared():
					self.has_curvature_info = False

					self.history_s = deque()
					self.history_y = deque()
					self.history_rho = deque()

				else:
					self.has_curvature_info = True
					rho = 1/self.curvature_condition
					self.history_rho.appendleft(rho)

				if len(self.history_s) > self.memory_vectors:
					self.history_s.pop()
					self.history_y.pop()
					self.history_rho.pop()


		# if not self.line_search_broken:
		# 	self.print_results()

		if self.verbose:
			print('')
			print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
			print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
			print('')
