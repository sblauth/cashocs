"""
Created on 01.04.20, 11:27

@author: sebastian
"""

import fenics
import numpy as np
from ..optimization_algorithm import OptimizationAlgorithm
from .pdas_inner_solvers.inner_gradient_descent import InnerGradientDescent
from .pdas_inner_solvers.inner_cg import InnerCG
from .pdas_inner_solvers.inner_lbfgs import InnerLBFGS
from .pdas_inner_solvers.inner_newton import InnerNewton



class PDAS(OptimizationAlgorithm):

	def __init__(self, optimization_problem):
		"""A primal dual active set method used to solve (control constrained) optimal control problems

		Parameters
		----------
		optimization_problem : adpack.optimization_problem.OptimizationProblem
			the OptimizationProblem object
		"""

		OptimizationAlgorithm.__init__(self, optimization_problem)

		self.idx_active_upper_prev = [np.array([]) for j in range(self.optimization_problem.control_dim)]
		self.idx_active_lower_prev = [np.array([]) for j in range(self.optimization_problem.control_dim)]
		self.initialized = False
		self.converged = False
		self.mu = [fenics.Function(self.optimization_problem.control_spaces[j]) for j in range(self.optimization_problem.control_dim)]
		self.shift_mult = self.config.getfloat('OptimizationRoutine', 'pdas_shift_mult')
		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')

		self.inner_pdas = self.config.get('OptimizationRoutine', 'inner_pdas')
		if self.inner_pdas == 'gradient_descent':
			self.inner_solver = InnerGradientDescent(optimization_problem)
		elif self.inner_pdas == 'cg':
			self.inner_solver = InnerCG(optimization_problem)
		elif self.inner_pdas == 'lbfgs':
			self.inner_solver = InnerLBFGS(optimization_problem)
		elif self.inner_pdas == 'newton':
			self.inner_solver = InnerNewton(optimization_problem)
		else:
			raise SystemExit('OptimizationRoutine.inner_pdas needs to be one of gradient_descent, lbfgs, cg, or newton.')



	def compute_active_inactive_sets(self):
		self.idx_active_lower = [(self.mu[j].vector()[:] + self.shift_mult*(self.optimization_problem.controls[j].vector()[:] - self.optimization_problem.control_constraints[j][0]) < 0).nonzero()[0]
								 for j in range(self.optimization_problem.control_dim)]
		self.idx_active_upper = [(self.mu[j].vector()[:] + self.shift_mult*(self.optimization_problem.controls[j].vector()[:] - self.optimization_problem.control_constraints[j][1]) > 0).nonzero()[0]
								 for j in range(self.optimization_problem.state_dim)]

		self.idx_active = [np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j])) for j in range(self.optimization_problem.control_dim)]
		[self.idx_active[j].sort() for j in range(self.optimization_problem.control_dim)]

		self.idx_inactive = [np.setdiff1d(np.arange(self.optimization_problem.control_spaces[j].dim()), self.idx_active[j]) for j in range(self.optimization_problem.control_dim)]

		if self.initialized:
			if all([np.array_equal(self.idx_active_upper[j], self.idx_active_upper_prev[j]) and np.array_equal(self.idx_active_lower[j], self.idx_active_lower_prev[j]) for j in range(self.optimization_problem.control_dim)]):
				self.converged = True

		self.idx_active_upper_prev = [self.idx_active_upper[j] for j in range(self.optimization_problem.control_dim)]
		self.idx_active_lower_prev = [self.idx_active_lower[j] for j in range(self.optimization_problem.control_dim)]
		self.initialized = True



	def run(self):
		self.iteration = 0
		self.relative_norm = 1.0

		### TODO: Check feasible initialization

		self.compute_active_inactive_sets()

		func_value = self.optimization_problem.reduced_cost_functional.compute()
		self.optimization_problem.state_problem.has_solution = True
		self.optimization_problem.gradient_problem.solve()
		norm_init = np.sqrt(self.optimization_problem.stationary_measure_squared())
		self.optimization_problem.adjoint_problem.has_solution = True

		print('Iteration: ' + str(self.iteration) + ' - Objective value: ' + format(func_value,'.3e') + ' - Gradient norm: ' + format(norm_init, '.3e'))

		while True:

			for j in range(len(self.controls)):
				self.controls[j].vector()[self.idx_active_lower[j]] = self.optimization_problem.control_constraints[j][0]
				self.controls[j].vector()[self.idx_active_upper[j]] = self.optimization_problem.control_constraints[j][1]


			self.inner_solver.run(self.idx_active)

			for j in range(len(self.controls)):
				self.mu[j].vector()[:] = -self.optimization_problem.gradients[j].vector()[:]
				self.mu[j].vector()[self.idx_inactive[j]] = 0.0

			self.iteration += 1

			func_value = self.inner_solver.line_search.objective_step
			norm = np.sqrt(self.optimization_problem.stationary_measure_squared())


			if self.iteration >= self.maximum_iterations:
				self.print_results()
				if self.soft_exit:
					print('Maximum number of iterations exceeded.')
					break
				else:
					raise SystemExit('Maximum number of iterations exceeded.')

			self.compute_active_inactive_sets()

			if self.converged:
				print('')
				print('Primal Dual Active Set Method Converged.')
				break

			print('Iteration: ' + str(self.iteration) + ' - Objective value: ' + format(func_value, '.3e') + ' - Gradient norm: ' + format(norm / norm_init, '.3e') + ' (rel)')

		if self.verbose:
			print('Statistics --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
			if self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton':
				print('           --- Sensitivity equations solved: ' + str(self.optimization_problem.unconstrained_hessian.no_sensitivity_solves))
			print('')
