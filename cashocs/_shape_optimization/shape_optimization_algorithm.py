# Copyright (C) 2020-2021 Sebastian Blauth
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

"""Blueprints for shape optimization algorithms.

"""

import json
import os
from pathlib import Path

import fenics

from .._loggers import info, error
from ..utils import write_out_mesh
from .._exceptions import NotConvergedError



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
		self.requires_remeshing = False
		self.remeshing_its = False
		self.has_curvature_info = False

		self.optimization_problem = optimization_problem
		self.form_handler = self.optimization_problem.form_handler
		self.state_problem = self.optimization_problem.state_problem
		self.config = self.state_problem.config
		self.adjoint_problem = self.optimization_problem.adjoint_problem

		self.shape_gradient_problem = self.optimization_problem.shape_gradient_problem
		self.gradient = self.shape_gradient_problem.gradient
		self.cost_functional = self.optimization_problem.reduced_cost_functional
		self.search_direction = fenics.Function(self.form_handler.deformation_space)

		if self.config.getboolean('Mesh', 'remesh', fallback=False):
			self.iteration = self.optimization_problem.temp_dict['OptimizationRoutine'].get('iteration_counter', 0)
		else:
			self.iteration = 0
		self.objective_value = 1.0
		self.gradient_norm_initial = 1.0
		self.relative_norm = 1.0
		self.stepsize = 1.0
		
		self.converged = False
		self.converged_reason = 0

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

		self.verbose = self.config.getboolean('Output', 'verbose', fallback=True)
		self.save_results = self.config.getboolean('Output', 'save_results', fallback=True)
		self.save_txt = self.config.getboolean('Output', 'save_txt', fallback=True)
		self.rtol = self.config.getfloat('OptimizationRoutine', 'rtol', fallback=1e-3)
		self.atol = self.config.getfloat('OptimizationRoutine', 'atol', fallback=0.0)
		self.maximum_iterations = self.config.getint('OptimizationRoutine', 'maximum_iterations', fallback=100)
		self.soft_exit = self.config.getboolean('OptimizationRoutine', 'soft_exit', fallback=False)
		self.save_pvd = self.config.getboolean('Output', 'save_pvd', fallback=False)
		self.save_pvd_adjoint = self.config.getboolean('Output', 'save_pvd_adjoint', fallback=False)
		self.save_pvd_gradient = self.config.getboolean('Output', 'save_pvd_gradient', fallback=False)
		self.result_dir = self.config.get('Output', 'result_dir', fallback='./')
		
		if not os.path.isdir(self.result_dir):
			Path(self.result_dir).mkdir(parents=True, exist_ok=True)
		
		if self.save_pvd:
			self.state_pvd_list = []
			for i in range(self.form_handler.state_dim):
				if self.form_handler.state_spaces[i].num_sub_spaces() > 0:
					self.state_pvd_list.append([])
					for j in range(self.form_handler.state_spaces[i].num_sub_spaces()):
						if self.optimization_problem.mesh_handler.do_remesh:
							self.state_pvd_list[i].append(fenics.File(self.result_dir + '/pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
																	  + '_state_' + str(i) + '_' + str(j) + '.pvd'))
						else:
							self.state_pvd_list[i].append(fenics.File(self.result_dir + '/pvd/state_' + str(i) + '_' + str(j) + '.pvd'))
				else:
					if self.optimization_problem.mesh_handler.do_remesh:
						self.state_pvd_list.append(fenics.File(self.result_dir + '/pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
															   + '_state_' + str(i) + '.pvd'))
					else:
						self.state_pvd_list.append(fenics.File(self.result_dir + '/pvd/state_' + str(i) + '.pvd'))

		if self.save_pvd_adjoint:
			self.adjoint_pvd_list = []
			for i in range(self.form_handler.state_dim):
				if self.form_handler.adjoint_spaces[i].num_sub_spaces() > 0:
					self.adjoint_pvd_list.append([])
					for j in range(self.form_handler.adjoint_spaces[i].num_sub_spaces()):
						if self.optimization_problem.mesh_handler.do_remesh:
							self.adjoint_pvd_list[i].append(fenics.File(self.result_dir + '/pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
																	  + '_adjoint_' + str(i) + '_' + str(j) + '.pvd'))
						else:
							self.adjoint_pvd_list[i].append(fenics.File(self.result_dir + '/pvd/adjoint_' + str(i) + '_' + str(j) + '.pvd'))
				else:
					if self.optimization_problem.mesh_handler.do_remesh:
						self.adjoint_pvd_list.append(fenics.File(self.result_dir + '/pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
															   + '_adjoint_' + str(i) + '.pvd'))
					else:
						self.adjoint_pvd_list.append(fenics.File(self.result_dir + '/pvd/adjoint_' + str(i) + '.pvd'))

		if self.save_pvd_gradient:
			if self.optimization_problem.mesh_handler.do_remesh:
				self.shape_gradient_pvd_file = fenics.File(self.result_dir + '/pvd/remesh_' + format(self.optimization_problem.temp_dict.get('remesh_counter', 0), 'd')
														   + '_shape_gradient.pvd' )

			else:
				self.shape_gradient_pvd_file = fenics.File(self.result_dir + '/pvd/shape_gradient.pvd')



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
			for i in range(self.form_handler.state_dim):
				if self.form_handler.state_spaces[i].num_sub_spaces() > 0:
					for j in range(self.form_handler.state_spaces[i].num_sub_spaces()):
						self.state_pvd_list[i][j] << (self.form_handler.states[i].sub(j, True), float(self.iteration))
				else:
					self.state_pvd_list[i] << (self.form_handler.states[i], float(self.iteration))

		if self.save_pvd_adjoint:
			for i in range(self.form_handler.state_dim):
				if self.form_handler.adjoint_spaces[i].num_sub_spaces() > 0:
					for j in range(self.form_handler.adjoint_spaces[i].num_sub_spaces()):
						self.adjoint_pvd_list[i][j] << (self.form_handler.adjoints[i].sub(j, True), float(self.iteration))
				else:
					self.adjoint_pvd_list[i] << (self.form_handler.adjoints[i], float(self.iteration))

		if self.save_pvd_gradient:
			self.shape_gradient_pvd_file << self.gradient, float(self.iteration)

		if self.verbose:
			print(output)

		if self.save_txt:
			if self.iteration == 0:
				with open(self.result_dir + 'history.txt', 'w') as file:
					file.write(output + '\n')
			else:
				with open(self.result_dir + 'history.txt', 'a') as file:
					file.write(output + '\n')
	


	def print_summary(self):
		"""Prints a summary of the (successful) optimization to console
		
		Returns
		-------
		None
		"""
		
		if self.verbose:
			print('\nStatistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)')
			print('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves) + '\n')

		if self.save_txt:
			with open(self.result_dir + 'history.txt', 'a') as file:
				file.write('\nStatistics --- Total iterations: ' + format(self.iteration, '4d') + ' --- Final objective value:  ' + format(self.objective_value, '.3e') +
				  ' --- Final gradient norm:  ' + format(self.relative_norm, '.3e') + ' (rel)\n')
				file.write('           --- State equations solved: ' + str(self.state_problem.number_of_solves) +
				  ' --- Adjoint equations solved: ' + str(self.adjoint_problem.number_of_solves) + '\n')


	
	def finalize(self):
		"""Saves the history of the optimization algorithm

		Returns
		-------
		None
		"""
		
		self.output_dict['initial_gradient_norm'] = self.gradient_norm_initial
		self.output_dict['state_solves'] = self.state_problem.number_of_solves
		self.output_dict['adjoint_solves'] = self.adjoint_problem.number_of_solves
		self.output_dict['iterations'] = self.iteration
		if self.save_results:
			with open(self.result_dir + '/history.json', 'w') as file:
				json.dump(self.output_dict, file)

		if self.optimization_problem.mesh_handler.save_optimized_mesh:
			write_out_mesh(self.optimization_problem.mesh_handler.mesh, self.optimization_problem.mesh_handler.gmsh_file, self.result_dir + '/optimized_mesh.msh')

		if self.optimization_problem.mesh_handler.do_remesh:
			if not self.config.getboolean('Debug', 'remeshing', fallback=False):
				os.system('rm -r ' + self.optimization_problem.temp_dir)
				os.system('rm -r ' + self.optimization_problem.mesh_handler.remesh_directory)


	
	def run(self):
		"""Blueprint run method, overriden by the actual solution algorithms

		Returns
		-------
		None
		"""

		pass
	
	
	
	def post_processing(self):
		"""Post processing of the solution algorithm
		
		Makes sure that the finalize method is called and that the output is written
		to files.
		
		Returns
		-------
		None
		"""
		
		if self.converged:
			self.print_results()
			self.print_summary()
			self.finalize()
			
		else:
			# maximum iterations reached
			if self.converged_reason == -1:
				self.print_results()
				if self.soft_exit:
					if self.verbose:
						print('Maximum number of iterations exceeded.')
					self.finalize()
				else:
					self.finalize()
					raise NotConvergedError('Optimization Algorithm', 'Maximum number of iterations were exceeded.')
			
			# Armijo line search failed
			elif self.converged_reason == -2:
				self.iteration -= 1
				if self.soft_exit:
					if self.verbose:
						print('Armijo rule failed.')
					self.finalize()
				else:
					self.finalize()
					raise NotConvergedError('Armijo line search', 'Failed to compute a feasible Armijo step.')
			
			# Mesh Quality is too low
			elif self.converged_reason == -3:
				self.iteration -= 1
				if self.optimization_problem.mesh_handler.do_remesh:
					info('Mesh Quality too low. Perform a remeshing operation.')
					self.optimization_problem.mesh_handler.remesh()
				else:
					if self.soft_exit:
						error('Mesh Quality is too low.')
						self.finalize()
					else:
						self.finalize()
						raise NotConvergedError('Optimization Algorithm', 'Mesh Quality is too low.')

			# Iteration for remeshing is the one exceeding the maximum number of iterations
			elif self.converged_reason == -4:
				if self.soft_exit:
					if self.verbose:
						print('Maximum number of iterations exceeded.')
					self.finalize()
				else:
					self.finalize()
					raise NotConvergedError('Optimization Algorithm', 'Maximum number of iterations were exceeded.')



	def nonconvergence(self):
		"""Checks for nonconvergence of the solution algorithm

		Returns
		-------
		 : boolean
			A flag which is True, when the algorithm did not converge
		"""

		if self.iteration >= self.maximum_iterations:
			self.converged_reason = -1
		if self.line_search_broken:
			self.converged_reason = -2
		if self.requires_remeshing:
			self.converged_reason = -3
		if self.remeshing_its:
			self.converged_reason = -4

		if self.converged_reason < 0:
			return True
		else:
			return False
