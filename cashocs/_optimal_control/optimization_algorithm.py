# Copyright (C) 2020 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Blueprint for the optimization algorithms.

"""

import json

import fenics



class OptimizationAlgorithm:
	"""Abstract class representing a optimization algorithm

	This is used for subclassing with the specific optimization methods
	later on.

	See Also
	--------
	methods.gradient_descent.GradientDescent
	methods.cg.CG
	methods.l_bfgs.LBFGS
	methods.newton.Newton
	methods.primal_dual_active_set_method.PDAS
	"""

	def __init__(self, optimization_problem):
		"""Initializes the optimization algorithm

		Defines common parameters used by all sub-classes.

		Parameters
		----------
		optimization_problem : cashocs._optimal_control.optimal_control_problem.OptimalControlProblem
			the OptimalControlProblem class as defined through the user
		"""

		self.line_search_broken = False
		self.has_curvature_info = False

		self.optimization_problem = optimization_problem
		self.form_handler = self.optimization_problem.form_handler
		self.state_problem = self.optimization_problem.state_problem
		self.config = self.optimization_problem.config

		self.adjoint_problem = self.optimization_problem.adjoint_problem

		self.gradient_problem = self.optimization_problem.gradient_problem
		self.gradients = self.optimization_problem.gradients
		self.controls = self.optimization_problem.controls
		self.controls_temp = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		self.projected_difference = [fenics.Function(V) for V in self.optimization_problem.control_spaces]
		self.search_directions = [fenics.Function(V) for V in self.optimization_problem.control_spaces]

		self.iteration = 0
		self.objective_value = 1.0
		self.gradient_norm_initial = 1.0
		self.relative_norm = 1.0
		self.stepsize = 1.0

		self.output_dict = dict()
		self.output_dict['cost_function_value'] = []
		self.output_dict['gradient_norm'] = []
		self.output_dict['stepsize'] = []

		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose', fallback=True)
		self.save_results = self.config.getboolean('OptimizationRoutine', 'save_results', fallback=True)
		self.rtol = self.config.getfloat('OptimizationRoutine', 'rtol', fallback=1e-2)
		self.atol = self.config.getfloat('OptimizationRoutine', 'atol', fallback=0.0)
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations', fallback=100)
		self.soft_exit = self.config.getboolean('OptimizationRoutine', 'soft_exit', fallback=False)
		self.save_pvd = self.config.getboolean('OptimizationRoutine', 'save_pvd', fallback=False)



		if self.save_pvd:
			self.state_pvd_list = []
			for i in range(self.form_handler.state_dim):
				if self.form_handler.state_spaces[i].num_sub_spaces() > 0:
					self.state_pvd_list.append([])
					for j in range(self.form_handler.state_spaces[i].num_sub_spaces()):
						self.state_pvd_list[i].append(fenics.File('./pvd/state_' + str(i) + '_' + str(j) + '.pvd'))
				else:
					self.state_pvd_list.append(fenics.File('./pvd/state_' + str(i) + '.pvd'))



	def print_results(self):
		"""Prints the current state of the optimization algorithm to the console.

		Returns
		-------
		None
		"""

		if self.iteration == 0:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.gradient_norm_initial, '.3e') + ' (abs) \n '
		else:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)    Step size:  ' + format(self.stepsize, '.3e')

		self.output_dict['cost_function_value'].append(self.objective_value)
		self.output_dict['gradient_norm'].append(self.relative_norm)
		self.output_dict['stepsize'].append(self.stepsize)

		if self.save_pvd:
			for i in range(self.form_handler.state_dim):
				if self.form_handler.state_spaces[i].num_sub_spaces() > 0:
					for j in range(self.form_handler.state_spaces[i].num_sub_spaces()):
						self.state_pvd_list[i][j] << self.form_handler.states[i].sub(j, True), self.iteration
				else:
					self.state_pvd_list[i] << self.form_handler.states[i], self.iteration

		if self.verbose:
			print(output)



	def finalize(self):
		"""Finalizes the solution algorithm.

		This saves the history of the optimization into the .json file.
		Called after the solver has finished.

		Returns
		-------
		None
		"""

		if self.verbose:
			print('')
			print('Statistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
			print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves))
			print('')

		self.output_dict['state_solves'] = self.state_problem.number_of_solves
		self.output_dict['adjoint_solves'] = self.adjoint_problem.number_of_solves
		self.output_dict['iterations'] = self.iteration
		if self.save_results:
			with open('./history.json', 'w') as file:
				json.dump(self.output_dict, file)



	def run(self):
		"""Blueprint for a print function

		This is overrriden by the specific optimization algorithms later on.

		Returns
		-------
		None
		"""

		pass
