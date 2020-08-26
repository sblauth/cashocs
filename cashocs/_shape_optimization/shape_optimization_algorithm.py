"""
Created on 15/06/2020, 08.00

@author: blauths
"""

import fenics
import json
import os
import sys



class ShapeOptimizationAlgorithm:
	"""Blueprint for a solution algorithm for shape optimization problems

	"""

	def __init__(self, optimization_problem):
		"""Parent class for the optimization methods implemented in cashocs.optimization.methods

		Parameters
		----------
		optimization_problem : cashocs.ShapeOptimizationProblem
			the optimization problem
		"""

		self.line_search_broken = False
		self.has_curvature_info = False

		self.optimization_problem = optimization_problem
		self.shape_form_handler = self.optimization_problem.shape_form_handler
		self.state_problem = self.optimization_problem.state_problem
		self.config = self.state_problem.config
		self.adjoint_problem = self.optimization_problem.adjoint_problem

		self.shape_gradient_problem = self.optimization_problem.shape_gradient_problem
		self.gradient = self.shape_gradient_problem.gradient
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		self.search_direction = fenics.Function(self.shape_form_handler.deformation_space)

		if self.config.getboolean('Mesh', 'remesh', fallback=False):
			self.iteration = self.optimization_problem.temp_dict['OptimizationRoutine'].get('iteration_counter', 0)
		else:
			self.iteration = 0
		self.objective_value = 1.0
		self.gradient_norm_initial = 1.0
		self.relative_norm = 1.0
		self.stepsize = 1.0

		self.output_dict = dict()
		try:
			self.output_dict['cost_function_value'] = self.optimization_problem.temp_dict['output_dict']['cost_function_value']
			self.output_dict['gradient_norm'] = self.optimization_problem.temp_dict['output_dict']['gradient_norm']
			self.output_dict['stepsize'] = self.optimization_problem.temp_dict['output_dict']['stepsize']
			self.output_dict['MeshQuality'] = self.optimization_problem.temp_dict['output_dict']['MeshQuality']
		except (TypeError, KeyError):
			self.output_dict['cost_function_value'] = []
			self.output_dict['gradient_norm'] = []
			self.output_dict['stepsize'] = []
			self.output_dict['MeshQuality'] = []

		self.verbose = self.config.getboolean('OptimizationRoutine', 'verbose', fallback=True)
		self.save_results = self.config.getboolean('OptimizationRoutine', 'save_results', fallback=True)
		self.rtol = self.config.getfloat('OptimizationRoutine', 'rtol', fallback=1e-2)
		self.atol = self.config.getfloat('OptimizationRoutine', 'atol', fallback=0.0)
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations', fallback=100)
		self.soft_exit = self.config.getboolean('OptimizationRoutine', 'soft_exit', fallback=False)
		self.save_pvd = self.config.getboolean('OptimizationRoutine', 'save_pvd', fallback=False)

		if self.save_pvd:
			self.state_pvd_list = []
			for i in range(self.shape_form_handler.state_dim):
				if self.shape_form_handler.state_spaces[i].num_sub_spaces() > 0:
					self.state_pvd_list.append([])
					for j in range(self.shape_form_handler.state_spaces[i].num_sub_spaces()):
						if self.config.getboolean('Mesh', 'remesh', fallback=False):
							self.state_pvd_list[i].append(fenics.File('./pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
																	  + '_state_' + str(i) + '_' + str(j) + '.pvd'))
						else:
							self.state_pvd_list[i].append(fenics.File('./pvd/state_' + str(i) + '_' + str(j) + '.pvd'))
				else:
					if self.config.getboolean('Mesh', 'remesh', fallback=False):
						self.state_pvd_list.append(fenics.File('./pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
															   + '_state_' + str(i) + '.pvd'))
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
					 '    Gradient norm:  ' + format(self.gradient_norm_initial, '.3e') + ' (abs)    Mesh Quality: ' + \
					 format(self.optimization_problem.mesh_handler.current_mesh_quality, '.2f') + ' (' + \
					 str(self.optimization_problem.mesh_handler.mesh_quality_measure) + ')' + ' \n '
		else:
			output = 'Iteration ' + format(self.iteration, '4d') + ' - Objective value:  ' + format(self.objective_value, '.3e') + \
					 '    Gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)    Mesh Quality: ' + \
					 format(self.optimization_problem.mesh_handler.current_mesh_quality, '.2f') + ' (' + \
					 str(self.optimization_problem.mesh_handler.mesh_quality_measure) + ')' + '    Step size:  ' + format(self.stepsize, '.3e')

		self.output_dict['cost_function_value'].append(self.objective_value)
		self.output_dict['gradient_norm'].append(self.relative_norm)
		self.output_dict['stepsize'].append(self.stepsize)
		self.output_dict['MeshQuality'].append(self.optimization_problem.mesh_handler.current_mesh_quality)

		if self.save_pvd:
			for i in range(self.shape_form_handler.state_dim):
				if self.shape_form_handler.state_spaces[i].num_sub_spaces() > 0:
					for j in range(self.shape_form_handler.state_spaces[i].num_sub_spaces()):
						self.state_pvd_list[i][j] << (self.shape_form_handler.states[i].sub(j, True), float(self.iteration))
				else:
					self.state_pvd_list[i] << (self.shape_form_handler.states[i], float(self.iteration))

		if self.verbose:
			print(output)



	def finalize(self):
		"""Saves the history of the optimization algorithm

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

		if self.config.getboolean('Mesh', 'remesh', fallback=False):
			os.system('rm -r ' + self.optimization_problem.temp_dir)



	def run(self):
		"""Blueprint run method, overriden by the actual solution algorithms

		Returns
		-------
		None
		"""

		pass
