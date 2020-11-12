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

"""Implementation of a shape optimization problem.

"""

import json
import os
import sys
import tempfile
import warnings

import fenics
import numpy as np
from ufl import replace
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree

from .methods import CG, GradientDescent, LBFGS
from .._exceptions import ConfigError, InputError, CashocsException
from .._forms import Lagrangian, ShapeFormHandler
from .._pde_problems import AdjointProblem, ShapeGradientProblem, StateProblem
from .._shape_optimization import ReducedShapeCostFunctional
from ..geometry import _MeshHandler
from ..optimization_problem import OptimizationProblem
from ..utils import _optimization_algorithm_configuration



class ShapeOptimizationProblem(OptimizationProblem):
	r"""A shape optimization problem.

	This class is used to define a shape optimization problem, and to solve
	it subsequently. For a detailed documentation, we refer to the :ref:`tutorial <tutorial_index>`.
	For easier input, when consider single (state or control) variables,
	these do not have to be wrapped into a list.
	Note, that in the case of multiple variables these have to be grouped into
	ordered lists, where state_forms, bcs_list, states, adjoints have to have
	the same order (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of ``y1``
	and so on.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states,
				 adjoints, boundaries, config=None, initial_guess=None,
				 ksp_options=None, adjoint_ksp_options=None):
		"""This is used to generate all classes and functionalities. First ensures
		consistent input, afterwards, the solution algorithm is initialized.

		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			The weak form of the state equation (user implemented). Can be either
			a single UFL form, or a (ordered) list of UFL forms.
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
			If this is ``None``, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form
			UFL form of the cost functional.
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
		boundaries : dolfin.cpp.mesh.MeshFunctionSizet
			:py:class:`fenics.MeshFunction` that indicates the boundary markers.
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
		"""

		OptimizationProblem.__init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config, initial_guess, ksp_options, adjoint_ksp_options)

		### Initialize the remeshing behavior, and a temp file
		self.do_remesh = self.config.getboolean('Mesh', 'remesh', fallback=False)
		self.temp_dict = None
		if self.do_remesh:

			if not os.path.isfile(os.path.realpath(sys.argv[0])):
				raise CashocsException('Not a valid configuration. The script has to be the first command line argument.')

			try:
				if __IPYTHON__:
					warnings.warn('You are running a shape optimization problem with remeshing from ipython. Rather run this using the python command.')
			except NameError:
				pass

			try:
				if not self.states[0].function_space().mesh()._cashocs_generator == 'config':
					raise InputError('cashocs.import_mesh', 'arg', 'You must specify a config file as input for remeshing.')
			except AttributeError:
				raise InputError('cashocs.import_mesh', 'arg', 'You must specify a config file as input for remeshing.')

			if not ('_cashocs_remesh_flag' in sys.argv):
				self.directory = os.path.dirname(os.path.realpath(sys.argv[0]))
				self.__clean_previous_temp_files()
				self.temp_dir = tempfile.mkdtemp(prefix='._cashocs_remesh_temp_', dir=self.directory)
				self.__change_except_hook()
				self.temp_dict = {'temp_dir' : self.temp_dir, 'gmsh_file' : self.config.get('Mesh', 'gmsh_file'),
								  'geo_file' : self.config.get('Mesh', 'geo_file'),
								  'OptimizationRoutine' : {'iteration_counter' : 0, 'gradient_norm_initial' : 0.0},
								  'output_dict' : {}}

			else:
				self.temp_dir = sys.argv[-1]
				self.__change_except_hook()
				with open(self.temp_dir + '/temp_dict.json', 'r') as file:
					self.temp_dict = json.load(file)

		### boundaries
		if boundaries.__module__ == 'dolfin.cpp.mesh' and type(boundaries).__name__ == 'MeshFunctionSizet':
			self.boundaries = boundaries
		else:
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem', 'boundaries', 'Not a valid type for boundaries.')

		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.shape_form_handler = ShapeFormHandler(self.lagrangian, self.bcs_list, self.states, self.adjoints,
												   self.boundaries, self.config, self.ksp_options, self.adjoint_ksp_options)
		self.mesh_handler = _MeshHandler(self)

		self.state_spaces = self.shape_form_handler.state_spaces
		self.adjoint_spaces = self.shape_form_handler.adjoint_spaces

		self.state_problem = StateProblem(self.shape_form_handler, self.initial_guess, self.temp_dict)
		self.adjoint_problem = AdjointProblem(self.shape_form_handler, self.state_problem, self.temp_dict)
		self.shape_gradient_problem = ShapeGradientProblem(self.shape_form_handler, self.state_problem, self.adjoint_problem)

		self.reduced_cost_functional = ReducedShapeCostFunctional(self.shape_form_handler, self.state_problem)

		self.gradient = self.shape_gradient_problem.gradient
		self.objective_value = 1.0



	def _erase_pde_memory(self):
		"""Resets the memory of the PDE problems so that new solutions are computed.

		This sets the value of has_solution to False for all relevant PDE problems,
		where memory is stored.

		Returns
		-------
		None
		"""

		self.mesh_handler.bbtree.build(self.mesh_handler.mesh)
		self.shape_form_handler.update_scalar_product()
		self.state_problem.has_solution = False
		self.adjoint_problem.has_solution = False
		self.shape_gradient_problem.has_solution = False



	def solve(self, algorithm=None, rtol=None, atol=None, max_iter=None):
		r"""Solves the optimization problem by the method specified in the config file.

		Parameters
		----------
		algorithm : str or None, optional
			Selects the optimization algorithm. Valid choices are
			``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
			``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
			for nonlinear conjugate gradient methods, and ``'lbfgs'`` or ``'bfgs'`` for
			limited memory BFGS methods. This overwrites the value specified
			in the config file. If this is ``None``, then the value in the
			config file is used. Default is ``None``.
		rtol : float or None, optional
			The relative tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is ``None``, the value from the config file is taken. Default
			is ``None``.
		atol : float or None, optional
			The absolute tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is ``None``, the value from the config file is taken. Default
			is ``None``.
		max_iter : int or None, optional
			The maximum number of iterations the optimization algorithm
			can carry out before it is terminated. Overwrites the value
			specified in the config file. If this is ``None``, the value from
			the config file is taken. Default is ``None``.

		Returns
		-------
		None

		Notes
		-----
		If either ``rtol`` or ``atol`` are specified as arguments to the solve
		call, the termination criterion changes to:

		  - a purely relative one (if only ``rtol`` is specified), i.e.,

		  .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

		  - a purely absolute one (if only ``atol`` is specified), i.e.,

		  .. math:: || \nabla J(u_K) || \leq \texttt{atol}.

		  - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

		  .. math:: || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol} || \nabla J(u_0) ||
		"""

		self.algorithm = _optimization_algorithm_configuration(self.config, algorithm)

		if (rtol is not None) and (atol is None):
			self.config.set('OptimizationRoutine', 'rtol', str(rtol))
			self.config.set('OptimizationRoutine', 'atol', str(0.0))
		elif (atol is not None) and (rtol is None):
			self.config.set('OptimizationRoutine', 'rtol', str(0.0))
			self.config.set('OptimizationRoutine', 'atol', str(atol))
		elif (atol is not None) and (rtol is not None):
			self.config.set('OptimizationRoutine', 'rtol', str(rtol))
			self.config.set('OptimizationRoutine', 'atol', str(atol))

		if max_iter is not None:
			self.config.set('OptimizationRoutine', 'maximum_iterations', str(max_iter))

		if self.algorithm == 'gradient_descent':
			self.solver = GradientDescent(self)
		elif self.algorithm == 'lbfgs':
			self.solver = LBFGS(self)
		elif self.algorithm == 'conjugate_gradient':
			self.solver = CG(self)
		elif self.algorithm == 'none':
			raise InputError('cashocs.OptimalControlProblem.solve', 'algorithm', 'You did not specify a solution algorithm in your config file. You have to specify one in the solve '
																				 'method. Needs to be one of \'gradient_descent\' (\'gd\'), \'lbfgs\' (\'bfgs\'), '
																				 'or \'conjugate_gradient\' (\'cg\').')
		else:
			raise ConfigError('OptimizationRoutine', 'algorithm', 'Not a valid input. Needs to be one '
																  'of \'gradient_descent\' (\'gd\'), \'lbfgs\' (\'bfgs\'), or \'conjugate_gradient\' (\'cg\').')

		self.solver.run()
		self.solver.post_processing()


	def __change_except_hook(self):
		"""Ensures that temp files are deleted when an exception occurs.

		This modifies the sys.excepthook command so that it also deletes temp files
		(only needed for remeshing)

		Returns
		-------
		None
		"""

		def custom_except_hook(exctype, value, traceback):
			# print('DEBUG: Caught the exception, deleting temp files')
			os.system('rm -r ' + self.temp_dir)
			sys.__excepthook__(exctype, value, traceback)

		sys.excepthook = custom_except_hook



	def __clean_previous_temp_files(self):

		for file in os.listdir(self.directory):
			if file.startswith('._cashocs_remesh_temp_'):
				os.system('rm -r ' + file)



	def compute_shape_gradient(self):
		"""Solves the Riesz problem to determine the shape gradient.

		This can be used for debugging, or code validation.
		The necessary solutions of the state and adjoint systems
		are carried out automatically.

		Returns
		-------
		dolfin.function.function.Function
			The shape gradient.
		"""

		self.shape_gradient_problem.solve()

		return self.gradient
	
	
	def supply_adjoint_forms(self, adjoint_forms, adjoint_bcs_list):
		"""Overrides the computed weak forms of the adjoint system
		
		This allows the user to specify their own weak forms of the problems and to use cashocs merely as
		a solver for solving the optimization problems.
		
		Parameters
		----------
		adjoint_forms : ufl.form.Form or list[ufl.form.Form]
			The UFL forms of the adjoint system(s)
		adjoint_bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of Dirichlet boundary conditions for the adjoint system(s)

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
			
		
		for idx, form in enumerate(mod_forms):
			if len(form.arguments()) == 2:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms', 'adjoint_forms',
								 'Do not use TrialFunction for the adjoints, but the actual Function you passed to th OptimalControlProblem.')
			elif len(form.arguments()) == 0:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms', 'adjoint_forms',
								 'The specified adjoint_forms must include a TestFunction object.')
			
			if not form.arguments()[0].ufl_function_space() == self.shape_form_handler.adjoint_spaces[idx]:
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_adjoint_forms', 'adjoint_forms',
								 'The TestFunction has to be chosen from the same space as the corresponding adjoint.')
		
		self.shape_form_handler.adjoint_picard_forms = mod_forms
		self.shape_form_handler.bcs_list_ad = mod_bcs_list

		# replace the adjoint function by a TrialFunction for internal use
		repl_forms = [replace(mod_forms[i], {self.adjoints[i] : self.shape_form_handler.trial_functions_adjoint[i]}) for i in range(self.state_dim)]
		self.shape_form_handler.adjoint_eq_forms = repl_forms
		
		self.shape_form_handler.adjoint_eq_lhs = []
		self.shape_form_handler.adjoint_eq_rhs = []

		for i in range(self.state_dim):
			a, L = fenics.system(self.shape_form_handler.adjoint_eq_forms[i])
			self.shape_form_handler.adjoint_eq_lhs.append(a)
			if L.empty():
				zero_form = fenics.inner(fenics.Constant(np.zeros(self.shape_form_handler.test_functions_adjoint[i].ufl_shape)),
										 self.shape_form_handler.test_functions_adjoint[i])*self.shape_form_handler.dx
				self.shape_form_handler.adjoint_eq_rhs.append(zero_form)
			else:
				self.shape_form_handler.adjoint_eq_rhs.append(L)
	
	
	
	def supply_shape_derivative(self, shape_derivative):
		"""Overrides the shape derivative of the reduced cost functional
		
		This allows users to implement their own shape derivative and use cashocs as a
		solver library only.
		
		Parameters
		----------
		shape_derivative : ufl.form.Form
			The shape_derivative of the reduced (!) cost functional w.r.t. controls

		Returns
		-------
		None
		"""
		try:
			if not shape_derivative.__module__ == 'ufl.form' and type(shape_derivative).__name__ == 'Form':
				raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative',
								 'shape_derivative', 'shape_derivative have to be a ufl form')
		except:
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative',
							 'shape_derivative', 'shape_derivative has to be a ufl form')
		
		if len(shape_derivative.arguments()) == 2:
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative',
							 'shape_derivative', 'Do not use TrialFunction for the shape_derivative.')
		elif len(shape_derivative.arguments()) == 0:
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative',
							 'shape_derivative', 'The specified shape_derivative must include a TestFunction object.')
		
		if not shape_derivative.arguments()[0].ufl_function_space().ufl_element() == self.shape_form_handler.deformation_space.ufl_element():
			raise InputError('cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative',
							 'shape_derivative', 'The TestFunction has to be chosen from the same space as the corresponding adjoint.')
		
		if not shape_derivative.arguments()[0].ufl_function_space() == self.shape_form_handler.deformation_space:
			shape_derivative = replace(shape_derivative, {shape_derivative.arguments()[0] : self.shape_form_handler.test_vector_field})
		
		if self.shape_form_handler.degree_estimation:
			estimated_degree = np.maximum(estimate_total_polynomial_degree(self.shape_form_handler.riesz_scalar_product),
											   estimate_total_polynomial_degree(shape_derivative))
			self.shape_form_handler.assembler = fenics.SystemAssembler(self.shape_form_handler.riesz_scalar_product, shape_derivative, self.shape_form_handler.bcs_shape,
													form_compiler_parameters={'quadrature_degree' : estimated_degree})
		else:
			try:
				self.shape_form_handler.assembler = fenics.SystemAssembler(self.shape_form_handler.riesz_scalar_product, shape_derivative, self.shape_form_handler.bcs_shape)
			except (AssertionError, ValueError):
				estimated_degree = np.maximum(estimate_total_polynomial_degree(self.shape_form_handler.riesz_scalar_product),
											   estimate_total_polynomial_degree(shape_derivative))
				self.shape_form_handler.assembler = fenics.SystemAssembler(self.shape_form_handler.riesz_scalar_product, shape_derivative, self.shape_form_handler.bcs_shape,
													form_compiler_parameters={'quadrature_degree' : estimated_degree})
	
	
	
	def supply_custom_forms(self, shape_derivative, adjoint_forms, adjoint_bcs_list):
		"""Overrides both adjoint system and shape derivative with user input.
		
		This allows the user to specify both the shape_derivative of the reduced cost functional
		and the corresponding adjoint system, and allows them to use cashocs as a solver.
		
		See Also
		--------
		supply_shape_derivative
		supply_adjoint_forms
		
		Parameters
		----------
		shape_derivative : ufl.form.Form
			The shape derivative of the reduced (!) cost functional
		adjoint_forms : ufl.form.Form or list[ufl.form.Form]
			The UFL forms of the adjoint system(s)
		adjoint_bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of Dirichlet boundary conditions for the adjoint system(s)

		Returns
		-------
		None
		"""
		
		self.supply_shape_derivative(shape_derivative)
		self.supply_adjoint_forms(adjoint_forms, adjoint_bcs_list)
	
	
	
	def get_vector_field(self):
		"""Returns the TestFunction for defining shape derivatives
		
		See Also
		--------
		supply_shape_derivative
		
		Returns
		-------
		 : dolfin.function.argument.Argument
		The TestFunction object
		"""
		
		return self.shape_form_handler.test_vector_field
