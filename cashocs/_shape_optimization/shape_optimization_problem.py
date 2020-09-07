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

"""
Created on 15/06/2020, 08.00

@author: blauths
"""

import json
import os
import sys
import tempfile
import warnings

from .methods import CG, GradientDescent, LBFGS
from .._exceptions import ConfigError, InputError, CashocsException
from .._forms import Lagrangian, ShapeFormHandler
from .._pde_problems import AdjointProblem, ShapeGradientProblem, StateProblem
from .._shape_optimization import ReducedShapeCostFunctional
from ..geometry import _MeshHandler
from ..optimization_problem import OptimizationProblem
from ..utils import _optimization_algorithm_configuration



class ShapeOptimizationProblem(OptimizationProblem):
	"""A shape optimization problem

	This class is used to define a shape optimization problem, and also to solve
	it subsequently. For a detailed documentation, see the examples in the "demos"
	folder. For easier input, when consider single (state or control) variables,
	these do not have to be wrapped into a list.
	Note, that in the case of multiple variables these have to be grouped into
	ordered lists, where state_forms, bcs_list, states, adjoints have to have
	the same order (i.e. [state1, state2, state3,...] and [adjoint1, adjoint2,
	adjoint3, ...], where adjoint1 is the adjoint of state1 and so on.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states,
				 adjoints, boundaries, config, initial_guess=None,
				 ksp_options=None, adjoint_ksp_options=None):
		"""Initializes the shape optimization problem

		This is used to generate all classes and functionalities. First ensures
		consistent input as the __init__ function is overloaded. Afterwards, the
		solution algorithm is initialized.

		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			the weak form of the state equation (user implemented). Can be either
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]]
				   or dolfin.fem.dirichletbc.DirichletBC or None
			the list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
			If this is None, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form
			UFL form of the cost functional
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the state variable(s), can either be a single fenics Function, or a (ordered) list of these
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the adjoint variable(s), can either be a single fenics Function, or a (ordered) list of these
		boundaries : dolfin.cpp.mesh.MeshFunctionSizet
			MeshFunction that indicates the boundary markers
		config : configparser.ConfigParser
			the config file for the problem, generated via cashocs.create_config(path_to_config)
		initial_guess : list[dolfin.function.function.Function], optional
			A list of functions that act as initial guess for the state variables, should be valid input for fenics.assign.
			(defaults to None (which means a zero initial guess))
		ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
			A list of strings corresponding to command line options for PETSc,
			used to solve the state systems. If this is None, then the direct solver
			mumps is used (default is None).
		adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
			A list of strings corresponding to command line options for PETSc,
			used to solve the adjoint systems. If this is None, then the same options
			as for the state systems are used (default is None).
		"""

		OptimizationProblem.__init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config, initial_guess, ksp_options, adjoint_ksp_options)

		### Initialize the remeshing behavior, and a temp file
		self.do_remesh = config.getboolean('Mesh', 'remesh', fallback=False)
		self.temp_dict = None
		if self.do_remesh:

			if not os.path.isfile(os.path.realpath(sys.argv[0])):
				raise CashocsException('Not a valid configuration. The script has to be the first command line argument.')

			try:
				if __IPYTHON__:
					warnings.warn('You are running a shape optimization problem with remeshing from ipython. Rather run this using the python command')
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
			'gradient_descent' ('gd'), 'conjugate_gradient' ('cg'),
			or 'lbfgs' ('bfgs'). This overwrites the value specified
			in the config file. If this is None, then the value in the
			config file is used. Default is None.
		rtol : float or None, optional
			The relative tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is None, the value from the config file is taken. Default
			is None.
		atol : float or None, optional
			The absolute tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is None, the value from the config file is taken. Default
			is None.
		max_iter : int or None, optional
			The maximum number of iterations the optimization algorithm
			can carry out before it is terminated. Overwrites the value
			specified in the config file. If this is None, the value from
			the config file is taken. Default is None.

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
		else:
			raise ConfigError('OptimizationRoutine', 'algorithm', 'Not a valid input. Needs to be one of \'gradient_descent\' (\'gd\'), \'lbfgs\' (\'bfgs\'), or \'conjugate_gradient\' (\'cg\').')

		self.solver.run()
		self.solver.finalize()



	def __change_except_hook(self):
		"""Ensures that temp files are deleted when an exception occurs.

		This modifies the sys.excepthook command so that it also deletes temp files
		(only needed for remeshing)

		Returns
		-------
		None
		"""

		def custom_except_hook(exctype, value, traceback):
			print('DEBUG: Caught the exception, deleting temp files')
			os.system('rm -r ' + self.temp_dir)
			sys.__excepthook__(exctype, value, traceback)

		sys.excepthook = custom_except_hook



	def __clean_previous_temp_files(self):

		for file in os.listdir(self.directory):
			if file.startswith('._cashocs_remesh_temp_'):
				os.system('rm -r ' + file)



	def compute_shape_gradient(self):
		"""Solves the Riesz problem to determine the gradient(s)

		This can be used for debugging, or code validation.
		The necessary solutions of the state and adjoint systems
		are carried out automatically.

		Returns
		-------
		dolfin.function.function.Function
			The shape gradient function
		"""

		self.shape_gradient_problem.solve()

		return self.gradient
