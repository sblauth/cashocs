"""
Created on 24/02/2020, 09.32

@author: blauths
"""

import fenics
import json



class OptimizationAlgorithm:
	
	def __init__(self, optimization_problem):
		"""Parent class for the optimization methods implemented in adpack.optimization.methods
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimizationProblem
			the OptimizationProblem class as defined through the user
		"""

		self.line_search_broken = False

		self.optimization_problem = optimization_problem
		self.form_handler = self.optimization_problem.form_handler
		self.state_problem = self.optimization_problem.state_problem
		self.config = self.state_problem.config
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

		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose')
		self.save_results = self.config.getboolean('OptimizationRoutine', 'save_results')
		# self.tolerance = self.config.getfloat('OptimizationRoutine', 'tolerance')
		self.rtol = self.config.getfloat('OptimizationRoutine', 'rtol')
		self.atol = self.config.getfloat('OptimizationRoutine', 'atol')
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations')
		self.soft_exit = self.config.getboolean('OptimizationRoutine', 'soft_exit')



	def print_results(self):
		"""Prints the current state of the optimization algorithm to the console.

		Returns
		-------
		None
			see method description

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

		if self.verbose:
			print(output)



	def finalize(self):
		if self.save_results:
			with open('./history.json', 'w') as file:
				json.dump(self.output_dict, file)



	def run(self):
		pass
