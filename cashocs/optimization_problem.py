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

"""Blueprints for the PDE constrained optimization problems.

This module is used to define the parent class for the optimization problems,
as many parameters and variables are common for optimal control and shape
optimization problems.
"""

import configparser
import json
import sys

import fenics
import numpy as np
from ufl import replace

from ._exceptions import InputError
from ._forms import Lagrangian, FormHandler
from ._pde_problems import StateProblem
from ._loggers import warning, info
from .utils import summation



class OptimizationProblem:
	"""Blueprint for an abstract PDE constrained optimization problem.

	This class performs the initialization of the shared input so that the rest
	of CASHOCS can use it directly. Additionally, it includes methods that
	can be used to compute the state and adjoint variables by solving the
	corresponding equations. This could be subclassed to generate custom
	optimization problems.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config=None,
				 initial_guess=None, ksp_options=None, adjoint_ksp_options=None, desired_weights=None):
		r"""Initializes the optimization problem.

		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			The weak form of the state equation (user implemented). Can be either
			a single UFL form, or a (ordered) list of UFL forms.
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of :py:class:`fenics.DirichletBC` objects describing Dirichlet (essential) boundary conditions.
			If this is ``None``, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form or list[ufl.form.Form]
			UFL form of the cost functional. Can also be a list of individual terms of the cost functional,
			which are scaled according to desired_weights.
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
		config : configparser.ConfigParser or None
			The config file for the problem, generated via :py:func:`cashocs.create_config`.
			Alternatively, this can also be ``None``, in which case the default configurations
			are used, except for the optimization algorithm. This has then to be specified
			in the :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
			default is ``None``.
		initial_guess : list[dolfin.function.function.Function], optional
			List of functions that act as initial guess for the state variables, should be valid input for :py:func:`fenics.assign`.
			Defaults to ``None``, which means a zero initial guess.
		ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
			A list of strings corresponding to command line options for PETSc,
			used to solve the state systems. If this is ``None``, then the direct solver
			mumps is used (default is ``None``).
		adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
			A list of strings corresponding to command line options for PETSc,
			used to solve the adjoint systems. If this is ``None``, then the same options
			as for the state systems are used (default is ``None``).
		desired_weights : list[int] or list[float] or None:
			A list which indicates the value of the associated term in the cost functional on
			the initial geometry in case a list of cost functions is supplied. If this is None,
			this defaults to multiplying all terms and adding them.

		Notes
		-----
		If one uses a single PDE constraint, the inputs can be the objects
		(UFL forms, functions, etc.) directly. In case multiple PDE constraints
		are present the inputs have to be put into (ordered) lists. The order of
		the objects depends on the order of the state variables, so that
		``state_forms[i]`` is the weak form of the PDE for ``states[i]`` with boundary
		conditions ``bcs_list[i]`` and corresponding adjoint state ``adjoints[i]``.

		See Also
		--------
		cashocs.OptimalControlProblem : Represents an optimal control problem.
		cashocs.ShapeOptimizationProblem : Represents a shape optimization problem.
		"""

		### Overloading, so that we do not have to use lists for a single state and a single control
		### state_forms
		try:
			if type(state_forms) == list and len(state_forms) > 0:
				for i in range(len(state_forms)):
					if state_forms[i].__module__=='ufl.form' and type(state_forms[i]).__name__=='Form':
						pass
					else:
						raise InputError('cashocs.optimization_problem.OptimizationProblem', 'state_forms', 'state_forms have to be ufl forms')
				self.state_forms = state_forms
			elif state_forms.__module__ == 'ufl.form' and type(state_forms).__name__ == 'Form':
				self.state_forms = [state_forms]
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'state_forms', 'state_forms have to be ufl forms')
		except:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'state_forms', 'state_forms have to be ufl forms')
		self.state_dim = len(self.state_forms)

		### bcs_list
		try:
			if bcs_list == [] or bcs_list is None:
				self.bcs_list = []
				for i in range(self.state_dim):
					self.bcs_list.append([])
			elif type(bcs_list) == list and len(bcs_list) > 0:
				if type(bcs_list[0]) == list:
					for i in range(len(bcs_list)):
						if type(bcs_list[i]) == list:
							pass
						else:
							raise InputError('cashocs.optimization_problem.OptimizationProblem', 'bcs_list', 'bcs_list has inconsistent types.')
					self.bcs_list = bcs_list

				elif bcs_list[0].__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list[0]).__name__ == 'DirichletBC':
					for i in range(len(bcs_list)):
						if bcs_list[i].__module__=='dolfin.fem.dirichletbc' and type(bcs_list[i]).__name__=='DirichletBC':
							pass
						else:
							raise InputError('cashocs.optimization_problem.OptimizationProblem', 'bcs_list', 'bcs_list has inconsistent types.')
					self.bcs_list = [bcs_list]
			elif bcs_list.__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list).__name__ == 'DirichletBC':
				self.bcs_list = [[bcs_list]]
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'bcs_list', 'Type of bcs_list is wrong.')
		except:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'bcs_list', 'Type of bcs_list is wrong.')

		### cost_functional_form
		self.use_cost_functional_list = False
		try:
			if type(cost_functional_form) == list:
				for term in cost_functional_form:
					if not term.__module__ == 'ufl.form' and type(term).__name__ == 'Form':
						raise InputError('cashocs.optimization_problem.OptimizationProblem', 'cost_functional_form', 'cost_functional_form has to be a ufl form or a list of ufl forms.')
					
				self.use_cost_functional_list = True
				self.cost_functional_list = cost_functional_form
				# generate a dummy cost_functional_form, which is overwritten in _scale_cost_functional
				self.cost_functional_form = summation([term for term in self.cost_functional_list])
					
			elif cost_functional_form.__module__ == 'ufl.form' and type(cost_functional_form).__name__ == 'Form':
				self.cost_functional_form = cost_functional_form
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'cost_functional_form', 'cost_functional_form has to be a ufl form.')
		except:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'cost_functional_form', 'Type of cost_functional_form is wrong.')

		### states
		try:
			if type(states) == list and len(states) > 0:
				for i in range(len(states)):
					if states[i].__module__ == 'dolfin.function.function' and type(states[i]).__name__ == 'Function':
						pass
					else:
						raise InputError('cashocs.optimization_problem.OptimizationProblem', 'states', 'states have to be fenics Functions.')

				self.states = states

			elif states.__module__ == 'dolfin.function.function' and type(states).__name__ == 'Function':
				self.states = [states]
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'states', 'Type of states is wrong.')
		except:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'states', 'Type of states is wrong.')

		### adjoints
		try:
			if type(adjoints) == list and len(adjoints) > 0:
				for i in range(len(adjoints)):
					if adjoints[i].__module__ == 'dolfin.function.function' and type(adjoints[i]).__name__ == 'Function':
						pass
					else:
						raise InputError('cashocs.optimization_problem.OptimizationProblem', 'adjoints', 'adjoints have to fenics Functions.')

				self.adjoints = adjoints

			elif adjoints.__module__ == 'dolfin.function.function' and type(adjoints).__name__ == 'Function':
				self.adjoints = [adjoints]
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'adjoints', 'Type of adjoints is wrong.')
		except:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'adjoints', 'Type of adjoints is wrong.')

		### config
		if config is None:
			self.config = configparser.ConfigParser()
			self.config.add_section('OptimizationRoutine')
			self.config.set('OptimizationRoutine', 'algorithm', 'none')
		else:
			if config.__module__ == 'configparser' and type(config).__name__ == 'ConfigParser':
				self.config = config
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'config', 'config has to be of configparser.ConfigParser type')

		### initial guess
		if initial_guess is None:
			self.initial_guess = initial_guess
		else:
			try:
				if type(initial_guess) == list:
					self.initial_guess = initial_guess
				elif initial_guess.__module__ == 'dolfin.function.function' and type(initial_guess).__name__ == 'Function':
					self.initial_guess = [initial_guess]
			except:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'initial_guess', 'initial guess has to be a list of functions')


		### ksp_options
		if ksp_options is None:
			self.ksp_options = []
			option = [
				['ksp_type', 'preonly'],
				['pc_type', 'lu'],
				['pc_factor_mat_solver_type', 'mumps'],
				['mat_mumps_icntl_24', 1]
			]

			for i in range(self.state_dim):
				self.ksp_options.append(option)

		elif type(ksp_options) == list and type(ksp_options[0]) == list and type(ksp_options[0][0]) == str:
			self.ksp_options = [ksp_options[:]]

		elif type(ksp_options) == list and type(ksp_options[0]) == list and type(ksp_options[0][0]) == list:
			self.ksp_options = ksp_options[:]

		else:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'ksp_options', 'Wrong input format for ksp_options.')



		### adjoint_ksp_options
		if adjoint_ksp_options is None:
			self.adjoint_ksp_options = self.ksp_options[:]

		elif type(adjoint_ksp_options) == list and type(adjoint_ksp_options[0]) == list and type(adjoint_ksp_options[0][0]) == str:
			self.adjoint_ksp_options = [adjoint_ksp_options[:]]

		elif type(adjoint_ksp_options) == list and type(adjoint_ksp_options[0]) == list and type(adjoint_ksp_options[0][0]) == list:
			self.adjoint_ksp_options = adjoint_ksp_options[:]

		else:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'adjoint_ksp_options', 'Wrong input format for adjoint_ksp_options.')

		### desired_weights
		if desired_weights is not None:
			if type(desired_weights) == list:
				for weight in desired_weights:
					if not (type(weight) == int or type(weight) == float):
						raise InputError('cashocs.optimization_problem.OptimizationProblem', 'desired_weights', 'desired_weights needs to be a list of numbers (int or float).')
				
				self.desired_weights = desired_weights
				
			else:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'desired_weights', 'desired_weights needs to be a list of numbers (int or float).')
		else:
			self.desired_weights = None


		if not len(self.bcs_list) == self.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'bcs_list', 'Length of states does not match.')
		if not len(self.states) == self.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'states', 'Length of states does not match.')
		if not len(self.adjoints) == self.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'adjoints', 'Length of states does not match.')

		if self.initial_guess is not None:
			if not len(self.initial_guess) == self.state_dim:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'initial_guess', 'Length of states does not match.')

		if not len(self.ksp_options) == self.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'ksp_options', 'Length of states does not match.')
		if not len(self.adjoint_ksp_options) == self.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'ksp_options', 'Length of states does not match.')
		
		if self.desired_weights is not None:
			try:
				if not len(self.cost_functional_list) == len(self.desired_weights):
					raise InputError('cashocs.optimization_problem.OptimizationProblem', 'desired_weights', 'Length of desired_weights and cost_functional does not match.')
			except:
				raise InputError('cashocs.optimization_problem.OptimizationProblem', 'desired_weights', 'Length of desired_weights and cost_functional does not match.')
			
		fenics.set_log_level(fenics.LogLevel.CRITICAL)

		self.state_problem = None
		self.adjoint_problem = None
		
		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.form_handler = None
		self.has_custom_adjoint = False
		self.has_custom_derivative = False
		
		self._scale_cost_functional()



	def compute_state_variables(self):
		"""Solves the state system.

		This can be used for debugging purposes and to validate the solver.
		Updates and overwrites the user input for the state variables.

		Returns
		-------
		None
		"""

		self.state_problem.solve()



	def compute_adjoint_variables(self):
		"""Solves the adjoint system.

		This can be used for debugging purposes and solver validation.
		Updates / overwrites the user input for the adjoint variables.
		The solve of the corresponding state system needed to determine
		the adjoints is carried out automatically.

		Returns
		-------
		None
		"""

		self.state_problem.solve()
		self.adjoint_problem.solve()
	
	
	
	def supply_adjoint_forms(self, adjoint_forms, adjoint_bcs_list):
		"""Overrides the computed weak forms of the adjoint system.
		
		This allows the user to specify their own weak forms of the problems and to use cashocs merely as
		a solver for solving the optimization problems.
		
		Parameters
		----------
		adjoint_forms : ufl.form.Form or list[ufl.form.Form]
			The UFL forms of the adjoint system(s).
		adjoint_bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of Dirichlet boundary conditions for the adjoint system(s).

		Returns
		-------
		None
		"""
		
		try:
			if type(adjoint_forms) == list and len(adjoint_forms) > 0:
				for i in range(len(adjoint_forms)):
					if adjoint_forms[i].__module__=='ufl.form' and type(adjoint_forms[i]).__name__=='Form':
						pass
					else:
						raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms',
										 'adjoint_forms', 'adjoint_forms have to be ufl forms')
				mod_forms = adjoint_forms
			elif adjoint_forms.__module__ == 'ufl.form' and type(adjoint_forms).__name__ == 'Form':
				mod_forms = [adjoint_forms]
			else:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms',
								 'adjoint_forms', 'adjoint_forms have to be ufl forms')
		except:
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms',
							 'adjoint_forms', 'adjoint_forms have to be ufl forms')
		
		
		
		try:
			if adjoint_bcs_list == [] or adjoint_bcs_list is None:
				mod_bcs_list = []
				for i in range(self.state_dim):
					mod_bcs_list.append([])
			elif type(adjoint_bcs_list) == list and len(adjoint_bcs_list) > 0:
				if type(adjoint_bcs_list[0]) == list:
					for i in range(len(adjoint_bcs_list)):
						if type(adjoint_bcs_list[i]) == list:
							pass
						else:
							raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms',
											 'adjoint_bcs_list', 'adjoint_bcs_list has inconsistent types.')
					mod_bcs_list = adjoint_bcs_list

				elif adjoint_bcs_list[0].__module__ == 'dolfin.fem.dirichletbc' and type(adjoint_bcs_list[0]).__name__ == 'DirichletBC':
					for i in range(len(adjoint_bcs_list)):
						if adjoint_bcs_list[i].__module__=='dolfin.fem.dirichletbc' and type(adjoint_bcs_list[i]).__name__=='DirichletBC':
							pass
						else:
							raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply adjoint_forms',
											 'adjoint_bcs_list', 'adjoint_bcs_list has inconsistent types.')
					mod_bcs_list = [adjoint_bcs_list]
			elif adjoint_bcs_list.__module__ == 'dolfin.fem.dirichletbc' and type(adjoint_bcs_list).__name__ == 'DirichletBC':
				mod_bcs_list = [[adjoint_bcs_list]]
			else:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms',
								 'adjoint_bcs_list', 'Type of adjoint_bcs_list is wrong.')
		except:
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms',
							 'adjoint_bcs_list', 'Type of adjoint_bcs_list is wrong.')
		
		if not len(mod_forms) == self.form_handler.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem.supply_adjoint_forms', 'adjoint_forms', 'Length of adjoint_forms does not match')
		if not len(mod_bcs_list) == self.form_handler.state_dim:
			raise InputError('cashocs.optimization_problem.OptimizationProblem.supply_adjoint_forms', 'adjoint_bcs_list', 'Length of adjoint_bcs_list does not match')
		
			
		
		for idx, form in enumerate(mod_forms):
			if len(form.arguments()) == 2:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms', 'adjoint_forms',
								 'Do not use TrialFunction for the adjoints, but the actual Function you passed to th OptimalControlProblem.')
			elif len(form.arguments()) == 0:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms', 'adjoint_forms',
								 'The specified adjoint_forms must include a TestFunction object.')
			
			if not form.arguments()[0].ufl_function_space() == self.form_handler.adjoint_spaces[idx]:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms', 'adjoint_forms',
								 'The TestFunction has to be chosen from the same space as the corresponding adjoint.')
		
		self.form_handler.adjoint_picard_forms = mod_forms
		self.form_handler.bcs_list_ad = mod_bcs_list

		# replace the adjoint function by a TrialFunction for internal use
		repl_forms = [replace(mod_forms[i], {self.adjoints[i] : self.form_handler.trial_functions_adjoint[i]}) for i in range(self.state_dim)]
		self.form_handler.adjoint_eq_forms = repl_forms
		
		self.form_handler.adjoint_eq_lhs = []
		self.form_handler.adjoint_eq_rhs = []

		for i in range(self.state_dim):
			a, L = fenics.system(self.form_handler.adjoint_eq_forms[i])
			self.form_handler.adjoint_eq_lhs.append(a)
			if L.empty():
				zero_form = fenics.inner(fenics.Constant(np.zeros(self.form_handler.test_functions_adjoint[i].ufl_shape)),
										 self.form_handler.test_functions_adjoint[i])*self.form_handler.dx
				self.form_handler.adjoint_eq_rhs.append(zero_form)
			else:
				self.form_handler.adjoint_eq_rhs.append(L)
		
		self.has_custom_adjoint = True
	
	
	
	def _check_for_custom_forms(self):
		"""Checks whether custom user forms are used and if they are compatible with the settings.
		
		Returns
		-------
		None
		"""
		
		if self.has_custom_adjoint and not self.has_custom_derivative:
			warning('You only supplied the adjoint system. This might lead to unexpected results.\n'
					'Consider also supplying the (shape) derivative of the reduced cost functional,'
					'or check your approach with the cashocs.verification module.')
		
		elif not self.has_custom_adjoint and self.has_custom_derivative:
			warning('You only supplied the derivative of the reduced cost functional. This might lead to unexpected results.\n'
					'Consider also supplying the adjoint system, '
					'or check your approach with the cashocs.verification module.')
		
		if self.algorithm == 'newton' and (self.has_custom_adjoint or self.has_custom_derivative):
			raise InputError('cashocs.optimization_problem.OptimizationProblem', 'solve', 'The usage of custom forms is not compatible with the Newton solver.'
																						  'Please do not supply custom forms if you want to use the Newton solver.')
		
		if self.use_cost_functional_list:
			info('You use the automatic scaling functionality of cashocs. This might lead to unexpected results if you try to scale the cost functional yourself.\n'
					'You can check your approach with the cashocs.verification module.')
	
	
	
	def _scale_cost_functional(self):
		
		if self.use_cost_functional_list:
			
			if self.desired_weights is not None:
				
				if not ('_cashocs_remesh_flag' in sys.argv):
					# Create dummy objects for adjoints, so that we can actually solve the state problem
					temp_form_handler = FormHandler(self.lagrangian, self.bcs_list, self.states, self.adjoints, self.config, self.ksp_options, self.adjoint_ksp_options)
					temp_state_problem = StateProblem(temp_form_handler, self.initial_guess)
					
					temp_state_problem.solve()
					self.initial_function_values = []
					for i in range(len(self.cost_functional_list)):
						val = fenics.assemble(self.cost_functional_list[i])
						
						if abs(val) <= 1e-15:
							val = 1.0
							info('Term ' + str(i) + ' of the cost functional vanishes for the initial iteration. Multiplying this term with the factor you supplied in desired_weights.')
							
						self.initial_function_values.append(val)
				
				else:
					temp_dir = sys.argv[-1]
					with open(temp_dir + '/temp_dict.json', 'r') as file:
						temp_dict = json.load(file)
					self.initial_function_values = temp_dict['initial_function_values']
				
				
				self.cost_functional_form = summation([fenics.Constant(abs(self.desired_weights[i] / self.initial_function_values[i]))*self.cost_functional_list[i]
													   for i in range(len(self.cost_functional_list))])
				
				self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
				
			else:
				self.cost_functional_form = summation([term for term in self.cost_functional_list])
